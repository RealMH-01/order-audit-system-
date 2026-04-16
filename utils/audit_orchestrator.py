"""
审核流程总调度器
协调文件解析、图片 OCR、单据审核、交叉比对的完整流程。
新增：Token 长度检测与分段处理、超时控制、细粒度进度反馈、取消审核支持。
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from utils.audit_engine import (
    build_audit_prompt,
    build_cross_check_prompt,
    parse_audit_result,
)
from utils.file_parser import parse_file
from utils.llm_client import (
    LLMError,
    call_llm,
    call_llm_with_image,
    IMAGE_OCR_PROMPT,
    TIMEOUT_NORMAL,
    TIMEOUT_DEEP_THINK,
)
from utils.token_utils import (
    estimate_tokens,
    smart_split_content,
    get_safe_token_limit,
)

logger = logging.getLogger(__name__)

# 从文件名推断单据类型的映射表
_DOC_TYPE_HINTS = {
    "ci": "商业发票CI",
    "invoice": "商业发票CI",
    "发票": "商业发票CI",
    "pl": "装箱单PL",
    "packing": "装箱单PL",
    "装箱": "装箱单PL",
    "booking": "托书Booking",
    "托书": "托书Booking",
    "生产通知": "生产通知单",
    "production": "生产通知单",
    "发货申请": "发货申请单",
    "shipping": "发货申请单",
}


def _guess_doc_type(filename: str) -> str:
    """根据文件名猜测单据类型。"""
    name_lower = filename.lower()
    for keyword, doc_type in _DOC_TYPE_HINTS.items():
        if keyword in name_lower:
            return doc_type
    return "待审核单据"


def run_full_audit(
    provider: str,
    api_key: str,
    po_data: Dict[str, Any],
    target_files_data: List[Dict[str, Any]],
    last_ticket_data: Optional[List[Dict[str, Any]]] = None,
    template_data: Optional[Dict[str, Any]] = None,
    other_refs_data: Optional[List[Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    deep_think: bool = False,
) -> Dict[str, Any]:
    """执行完整审核流程。

    Args:
        provider: 模型提供商名称。
        api_key: API 密钥。
        po_data: PO 文件的解析结果 dict。
        target_files_data: 待审核文件的解析结果 dict 列表。
        last_ticket_data: 上一票文件的解析结果 dict 列表（可选）。
        template_data: 标准模板的解析结果 dict（可选）。
        other_refs_data: 其他参考文件的解析结果 dict 列表（可选）。
        progress_callback: 进度回调函数，接收字符串参数。
        cancel_check: 取消检查函数，返回 True 表示用户要求取消。
        deep_think: 是否启用深度思考模式。

    Returns:
        {
            "per_file_results": {
                "文件名1": {审核结果字典},
                "文件名2": {审核结果字典},
            },
            "cross_check_result": {交叉比对结果字典} 或 None,
            "errors": ["无法审核的文件及原因列表"],
            "token_warning": "",  # Token 长度警告信息
            "cancelled": False,   # 是否被用户取消
        }
    """

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)

    def _is_cancelled() -> bool:
        if cancel_check and cancel_check():
            return True
        return False

    result: Dict[str, Any] = {
        "per_file_results": {},
        "cross_check_result": None,
        "errors": [],
        "token_warning": "",
        "cancelled": False,
    }

    # ==========================================================
    # 步骤 1：处理图片 OCR
    # ==========================================================
    _progress("正在准备审核数据...")

    if _is_cancelled():
        result["cancelled"] = True
        return result

    # 处理参考图片的 OCR
    other_refs_texts: list[str] = []
    if other_refs_data:
        images_to_ocr = [
            d for d in other_refs_data if d.get("is_image") and d.get("image_base64")
        ]
        if images_to_ocr:
            _progress(f"正在识别截图内容...（共 {len(images_to_ocr)} 张图片）")
            for img in images_to_ocr:
                if _is_cancelled():
                    result["cancelled"] = True
                    return result

                fname = img.get("filename", "未知图片")
                _progress(f"正在识别: {fname}")
                try:
                    ocr_text = call_llm_with_image(
                        provider, api_key, IMAGE_OCR_PROMPT, img["image_base64"]
                    )
                    img["content"] = ocr_text
                    other_refs_texts.append(f"[{fname}]\n{ocr_text}")
                except LLMError as e:
                    err_msg = f"图片 {fname} 识别失败: {e.message}"
                    result["errors"].append(err_msg)
                    _progress(f"⚠️ {err_msg}")

        # 非图片参考文件的文字也加入
        for d in other_refs_data:
            if not d.get("is_image") and d.get("content") and d.get("success"):
                other_refs_texts.append(
                    f"[{d.get('filename', '参考文件')}]\n{d['content']}"
                )

    # ==========================================================
    # 步骤 1.2：扫描件PDF自动OCR
    # ==========================================================


        # DeepSeek 不支持图片识别，检测到扫描件时提前终止并提示
    has_scanned_pdf = (
        (po_data.get("is_scanned_pdf") and po_data.get("pdf_page_images"))
        or any(t.get("is_scanned_pdf") and t.get("pdf_page_images") for t in target_files_data)
    )
    if has_scanned_pdf and provider.lower().strip() in ("deepseek",):
        error_msg = (
            "检测到上传的PDF为扫描件（图片型PDF），需要使用AI图片识别（OCR）来提取文字。"
            "但 DeepSeek API 目前不支持图片识别功能。"
            "请前往左侧边栏将大模型切换为「智谱GLM」，然后重新开始审核。"
        )
        _progress("❌ " + error_msg)
        result["errors"].append(error_msg)
        return result

    
    # --- 处理 PO 扫描件 ---
    if po_data.get("is_scanned_pdf") and po_data.get("pdf_page_images"):
        page_images = po_data["pdf_page_images"]
        total_pages = len(page_images)
        _progress(f"检测到PO为扫描件PDF，正在进行AI-OCR识别...（共 {total_pages} 页）")
        ocr_parts: list[str] = []
        for pg_idx, page_b64 in enumerate(page_images, start=1):
            if _is_cancelled():
                result["cancelled"] = True
                _progress("⚠️ 审核已被用户取消")
                return result

            _progress(f"正在识别PO扫描件第 {pg_idx}/{total_pages} 页...")
            try:
                ocr_text = call_llm_with_image(
                    provider, api_key, IMAGE_OCR_PROMPT, page_b64
                )
                ocr_parts.append(f"{'='*20} 第 {pg_idx} 页 {'='*20}\n{ocr_text}")
            except LLMError as e:
                err_msg = f"PO扫描件第 {pg_idx} 页识别失败: {e.message}"
                result["errors"].append(err_msg)
                _progress(f"⚠️ {err_msg}")
                ocr_parts.append(f"{'='*20} 第 {pg_idx} 页 {'='*20}\n[识别失败]")

        # 将 OCR 结果写回 po_data，供后续审核使用
        po_data["content"] = "\n\n".join(ocr_parts)
        _progress("✅ PO扫描件OCR识别完成")

    # --- 处理待审核文件中的扫描件 ---
    for target in target_files_data:
        if _is_cancelled():
            result["cancelled"] = True
            _progress("⚠️ 审核已被用户取消")
            return result

        if not target.get("is_scanned_pdf") or not target.get("pdf_page_images"):
            continue

        t_fname = target.get("filename", "未知文件")
        t_page_images = target["pdf_page_images"]
        t_total_pages = len(t_page_images)
        _progress(f"检测到 {t_fname} 为扫描件PDF，正在进行AI-OCR识别...（共 {t_total_pages} 页）")
        t_ocr_parts: list[str] = []
        all_failed = True
        for pg_idx, page_b64 in enumerate(t_page_images, start=1):
            if _is_cancelled():
                result["cancelled"] = True
                _progress("⚠️ 审核已被用户取消")
                return result

            _progress(f"正在识别 {t_fname} 第 {pg_idx}/{t_total_pages} 页...")
            try:
                ocr_text = call_llm_with_image(
                    provider, api_key, IMAGE_OCR_PROMPT, page_b64
                )
                t_ocr_parts.append(f"{'='*20} 第 {pg_idx} 页 {'='*20}\n{ocr_text}")
                all_failed = False
            except LLMError as e:
                err_msg = f"{t_fname} 第 {pg_idx} 页识别失败: {e.message}"
                result["errors"].append(err_msg)
                _progress(f"⚠️ {err_msg}")
                t_ocr_parts.append(f"{'='*20} 第 {pg_idx} 页 {'='*20}\n[识别失败]")

        # 将 OCR 结果写回 target，供后续审核使用
        target["content"] = "\n\n".join(t_ocr_parts)
        if all_failed:
            target["success"] = False
        else:
            target["success"] = True
        _progress(f"✅ {t_fname} 扫描件OCR识别完成")

    # 准备各文本
    po_text = po_data.get("content", "")
    template_text = (
        template_data.get("content", "") if template_data and template_data.get("success") else None
    )
    last_ticket_text = None
    if last_ticket_data:
        last_parts = []
        for d in last_ticket_data:
            if d.get("content") and d.get("success"):
                last_parts.append(d["content"])
        if last_parts:
            last_ticket_text = "\n\n---\n\n".join(last_parts)

    # ==========================================================
    # 步骤 1.5：Token 长度检测与智能分段处理
    # ==========================================================
    _progress("正在检测内容长度...")

    # 收集所有辅助文本
    auxiliary_texts = []
    if last_ticket_text:
        auxiliary_texts.append(last_ticket_text)
    if template_text:
        auxiliary_texts.append(template_text)
    auxiliary_texts.extend(other_refs_texts)

    # 对每个待审核文件进行 token 预检
    token_warning_issued = False
    for target in target_files_data:
        target_content = target.get("content", "")
        if not target_content or not target.get("success"):
            continue

        po_proc, target_proc, aux_proc, was_truncated = smart_split_content(
            po_text=po_text,
            target_text=target_content,
            other_texts=auxiliary_texts,
            provider=provider,
        )

        if was_truncated and not token_warning_issued:
            token_warning_issued = True
            warning_msg = (
                "⚠️ 文件内容较长，已自动优化处理，审核结果可能不如短文件精确。"
                "建议减少单次上传文件数量或拆分较长的文件。"
            )
            result["token_warning"] = warning_msg
            _progress(warning_msg)

    # ==========================================================
    # 步骤 2：逐份审核每个待审核文件
    # ==========================================================
    successful_targets: list[dict] = []  # 用于后续交叉比对
    total_files = len(target_files_data)

    for idx, target in enumerate(target_files_data, 1):
        if _is_cancelled():
            result["cancelled"] = True
            _progress("⚠️ 审核已被用户取消")
            return result

        fname = target.get("filename", f"文件{idx}")
        target_content = target.get("content", "")
        target_type = _guess_doc_type(fname)

        if not target_content or not target.get("success"):
            err_msg = f"{fname}: 文件解析失败，无法审核"
            result["errors"].append(err_msg)
            continue

        start_time = time.time()
        _progress(f"正在审核第 {idx}/{total_files} 份文件：{fname}...")

        # Token 智能分段处理
        po_processed, target_processed, aux_processed, _ = smart_split_content(
            po_text=po_text,
            target_text=target_content,
            other_texts=auxiliary_texts,
            provider=provider,
        )

        # 重建辅助文本
        last_ticket_processed = None
        template_processed = None
        other_refs_processed = []
        aux_idx = 0
        if last_ticket_text and aux_idx < len(aux_processed):
            last_ticket_processed = aux_processed[aux_idx]
            aux_idx += 1
        if template_text and aux_idx < len(aux_processed):
            template_processed = aux_processed[aux_idx]
            aux_idx += 1
        if aux_idx < len(aux_processed):
            other_refs_processed = aux_processed[aux_idx:]

        # 构造 prompt
        messages = build_audit_prompt(
            po_text=po_processed,
            target_text=target_processed,
            target_type=target_type,
            last_ticket_text=last_ticket_processed,
            template_text=template_processed,
            other_refs=other_refs_processed if other_refs_processed else None,
            deep_think=deep_think,
        )

        # 计算已耗时并更新进度
        elapsed = time.time() - start_time
        _progress(
            f"正在审核第 {idx}/{total_files} 份文件：{fname}"
            f"（已耗时 {int(elapsed)} 秒）"
        )

        # 调用大模型（含自动重试）
        audit_result = _call_and_parse(
            provider, api_key, messages, fname, result["errors"],
            deep_think=deep_think,
            progress_callback=lambda msg, f=fname, i=idx, t=total_files, st=start_time: _progress(
                f"正在审核第 {i}/{t} 份文件：{f}（已耗时 {int(time.time() - st)} 秒）— {msg}"
            ),
        )

        elapsed_final = time.time() - start_time
        if audit_result is not None:
            # 将原文文本附加到结果中，供报告生成使用
            audit_result["original_text"] = target_content
            result["per_file_results"][fname] = audit_result
            successful_targets.append(
                {"type": target_type, "content": target_content}
            )
            _progress(
                f"✅ {fname} 审核完成（耗时 {int(elapsed_final)} 秒）"
            )
        else:
            _progress(
                f"❌ {fname} 审核失败（耗时 {int(elapsed_final)} 秒）"
            )

    # ==========================================================
    # 步骤 3：交叉比对（仅当有多份待审核文件成功时）
    # ==========================================================
    if _is_cancelled():
        result["cancelled"] = True
        _progress("⚠️ 审核已被用户取消")
        return result

    if len(successful_targets) >= 2:
        _progress("正在进行单据间交叉比对...")
        cross_start = time.time()
        cross_messages = build_cross_check_prompt(successful_targets)
        cross_result = _call_and_parse(
            provider, api_key, cross_messages, "交叉比对", result["errors"],
            deep_think=deep_think,
        )
        cross_elapsed = time.time() - cross_start
        result["cross_check_result"] = cross_result
        _progress(f"✅ 交叉比对完成（耗时 {int(cross_elapsed)} 秒）")

    _progress("审核完成！")
    return result


def _call_and_parse(
    provider: str,
    api_key: str,
    messages: List[Dict],
    file_label: str,
    errors: list,
    max_retries: int = 2,
    deep_think: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """调用大模型并解析 JSON 结果，失败时自动重试。

    Args:
        provider: 模型提供商。
        api_key: API 密钥。
        messages: 完整的 messages 列表。
        file_label: 用于日志/错误信息的文件标签。
        errors: 错误列表，失败时追加错误信息。
        max_retries: 最大尝试次数。
        deep_think: 是否使用深度思考模式。
        progress_callback: 进度回调函数。

    Returns:
        解析后的审核结果字典，或 None。
    """
    for attempt in range(1, max_retries + 1):
        try:
            if progress_callback and attempt > 1:
                progress_callback(f"第 {attempt} 次尝试...")

            llm_response = call_llm(
                provider, api_key, messages,
                temperature=0.1,
                deep_think=deep_think,
            )
            parsed = parse_audit_result(llm_response)
            if parsed is not None:
                return parsed

            # 解析失败但有回复，如果还有重试机会
            if attempt < max_retries:
                logger.warning(
                    "[%s] 第%d次尝试: JSON解析失败，准备重试", file_label, attempt
                )
                # 追加一条重试消息
                messages = messages + [
                    {"role": "assistant", "content": llm_response},
                    {
                        "role": "user",
                        "content": (
                            "你的回复无法被解析为JSON格式。"
                            "请严格按照要求的JSON格式重新输出结果，"
                            "不要包含任何JSON以外的文字。"
                        ),
                    },
                ]
                continue
            else:
                err_msg = f"{file_label}: AI返回结果格式异常，无法解析"
                errors.append(err_msg)
                logger.error("[%s] JSON解析最终失败: %s", file_label, llm_response[:300])
                return None

        except LLMError as e:
            if attempt < max_retries:
                logger.warning(
                    "[%s] 第%d次尝试失败: %s，准备重试",
                    file_label,
                    attempt,
                    e.message,
                )
                continue
            err_msg = f"{file_label}: {e.message}"
            errors.append(err_msg)
            return None

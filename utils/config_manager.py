"""
配置管理模块
使用 st.session_state 管理用户设置，仅当前会话有效。
适配 Streamlit Community Cloud 部署（无本地持久化）。
"""

import streamlit as st

# ============================================================
# 默认值常量
# ============================================================
DEFAULT_MODEL = "DeepSeek"
MODEL_OPTIONS = ["DeepSeek", "智谱GLM"]

# session_state 键名常量
KEY_DISCLAIMER_ACCEPTED = "disclaimer_accepted"
KEY_DISCLAIMER_SKIP = "disclaimer_skip"
KEY_DISCLAIMER_CONFIRM_STEP = "disclaimer_confirm_step"
KEY_SELECTED_MODEL = "selected_model"
KEY_API_KEY = "api_key"
KEY_DEEP_THINK = "deep_think_enabled"
KEY_AUDIT_HISTORY = "audit_history"
KEY_CANCEL_AUDIT = "cancel_audit"
KEY_TOKEN_WARNING = "token_warning"


# ============================================================
# 初始化
# ============================================================
def init_session_state() -> None:
    """初始化所有 session_state 默认值（幂等操作）。"""
    defaults = {
        KEY_DISCLAIMER_ACCEPTED: False,  # 用户是否已通过免责声明
        KEY_DISCLAIMER_SKIP: False,      # 用户是否勾选"不再提示"
        KEY_DISCLAIMER_CONFIRM_STEP: "initial",  # 免责声明流程阶段
        KEY_SELECTED_MODEL: DEFAULT_MODEL,
        KEY_API_KEY: "",
        KEY_DEEP_THINK: False,           # DeepSeek 深度思考模式
        KEY_AUDIT_HISTORY: [],           # 审核历史记录列表
        KEY_CANCEL_AUDIT: False,         # 取消审核标志
        KEY_TOKEN_WARNING: "",           # Token 长度警告信息
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# 免责声明相关
# ============================================================
def is_disclaimer_accepted() -> bool:
    """用户是否已通过免责声明。"""
    return st.session_state.get(KEY_DISCLAIMER_ACCEPTED, False)


def accept_disclaimer() -> None:
    """标记免责声明已接受。"""
    st.session_state[KEY_DISCLAIMER_ACCEPTED] = True


def set_disclaimer_skip(skip: bool) -> None:
    """设置是否跳过免责声明。"""
    st.session_state[KEY_DISCLAIMER_SKIP] = skip


def is_disclaimer_skip() -> bool:
    """获取是否跳过免责声明。"""
    return st.session_state.get(KEY_DISCLAIMER_SKIP, False)


def get_disclaimer_step() -> str:
    """获取免责声明流程当前阶段。"""
    return st.session_state.get(KEY_DISCLAIMER_CONFIRM_STEP, "initial")


def set_disclaimer_step(step: str) -> None:
    """设置免责声明流程阶段。
    阶段：initial -> confirming -> done
    """
    st.session_state[KEY_DISCLAIMER_CONFIRM_STEP] = step


def reset_disclaimer() -> None:
    """重置免责声明状态，使其重新弹出。"""
    st.session_state[KEY_DISCLAIMER_ACCEPTED] = False
    st.session_state[KEY_DISCLAIMER_SKIP] = False
    st.session_state[KEY_DISCLAIMER_CONFIRM_STEP] = "initial"


# ============================================================
# 模型 & API 密钥
# ============================================================
def get_selected_model() -> str:
    return st.session_state.get(KEY_SELECTED_MODEL, DEFAULT_MODEL)


def set_selected_model(model: str) -> None:
    st.session_state[KEY_SELECTED_MODEL] = model


def get_api_key() -> str:
    return st.session_state.get(KEY_API_KEY, "")


def set_api_key(key: str) -> None:
    st.session_state[KEY_API_KEY] = key


# ============================================================
# 深度思考模式
# ============================================================
def is_deep_think_enabled() -> bool:
    """获取是否启用深度思考模式。"""
    return st.session_state.get(KEY_DEEP_THINK, False)


def set_deep_think_enabled(enabled: bool) -> None:
    """设置深度思考模式开关。"""
    st.session_state[KEY_DEEP_THINK] = enabled


# ============================================================
# 审核取消控制
# ============================================================
def is_audit_cancelled() -> bool:
    """获取是否已取消审核。"""
    return st.session_state.get(KEY_CANCEL_AUDIT, False)


def set_cancel_audit(cancel: bool) -> None:
    """设置取消审核标志。"""
    st.session_state[KEY_CANCEL_AUDIT] = cancel


# ============================================================
# Token 警告
# ============================================================
def get_token_warning() -> str:
    """获取 Token 长度警告信息。"""
    return st.session_state.get(KEY_TOKEN_WARNING, "")


def set_token_warning(warning: str) -> None:
    """设置 Token 长度警告信息。"""
    st.session_state[KEY_TOKEN_WARNING] = warning

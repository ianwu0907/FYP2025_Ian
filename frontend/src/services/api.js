/**
 * API 客户端
 * 处理与后端 API 的所有通信
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 分钟超时
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API] Response:`, response.data);
    return response;
  },
  (error) => {
    console.error('[API] Response error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const api = {
  /**
   * 上传文件
   * @param {File} file - 要上传的文件
   * @returns {Promise<Object>} 上传响应（包含 session_id 和预览）
   */
  uploadFile: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    return response.data;
  },

  /**
   * 开始标准化处理
   * @param {string} sessionId - 会话 ID
   * @param {Object} configOverrides - 配置覆盖
   * @returns {Promise<Object>} 任务响应（包含 task_id）
   */
  startNormalization: async (sessionId, configOverrides = {}) => {
    const response = await apiClient.post('/normalize', {
      session_id: sessionId,
      config_overrides: configOverrides,
    });

    return response.data;
  },

  /**
   * 获取任务状态
   * @param {string} taskId - 任务 ID
   * @returns {Promise<Object>} 任务状态
   */
  getTaskStatus: async (taskId) => {
    const response = await apiClient.get(`/normalize/status/${taskId}`);
    return response.data;
  },

  /**
   * 获取会话信息
   * @param {string} sessionId - 会话 ID
   * @returns {Promise<Object>} 会话信息
   */
  getSessionInfo: async (sessionId) => {
    const response = await apiClient.get(`/upload/session/${sessionId}`);
    return response.data;
  },

  /**
   * 下载标准化结果
   * @param {string} taskId - 任务 ID
   * @returns {Promise<Blob>} 文件 Blob
   */
  downloadResult: async (taskId) => {
    const response = await apiClient.get(`/normalize/download/${taskId}`, {
      responseType: 'blob', // 重要：指定响应类型为 blob
    });
    return response.data;
  },
};

export default api;

/**
 * useNormalizer Hook
 * 管理整个标准化流程的状态和逻辑
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { api } from '../services/api';

export const useNormalizer = () => {
  const [sessionId, setSessionId] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, uploaded, processing, completed, error
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploadedFileInfo, setUploadedFileInfo] = useState(null);

  const pollingIntervalRef = useRef(null);
  const pollingErrorCountRef = useRef(0); // 跟踪连续轮询错误次数

  /**
   * 上传文件
   */
  const uploadFile = useCallback(async (file) => {
    try {
      setStatus('uploading');
      setError(null);
      setLogs((prev) => [...prev, `Starting file upload: ${file.name}`]);

      const response = await api.uploadFile(file);

      setSessionId(response.session_id);
      setUploadedFileInfo(response);
      setStatus('uploaded');
      setLogs((prev) => [...prev, `File uploaded successfully! Session ID: ${response.session_id}`]);
      setLogs((prev) => [...prev, `File preview: ${response.preview.shape[0]} rows × ${response.preview.shape[1]} columns`]);

      return response;
    } catch (err) {
      console.error('Upload error:', err);
      setStatus('error');
      setError(err.response?.data?.detail || err.message || 'Upload failed');
      setLogs((prev) => [...prev, `❌ Upload failed: ${err.response?.data?.detail || err.message}`]);
      throw err;
    }
  }, []);

  /**
   * 开始标准化
   */
  const startNormalization = useCallback(async (configOverrides = {}) => {
    try {
      setStatus('processing');
      setProgress(0);
      setError(null);
      setLogs((prev) => [...prev, 'Starting normalization...']);

      const response = await api.startNormalization(sessionId, configOverrides);
      setTaskId(response.task_id);
      setLogs((prev) => [...prev, `Task created: ${response.task_id}`]);

      // 开始轮询任务状态
      startPolling(response.task_id);

      return response;
    } catch (err) {
      console.error('Start normalization error:', err);
      setStatus('error');
      setError(err.response?.data?.detail || err.message || 'Failed to start');
      setLogs((prev) => [...prev, `❌ Failed to start: ${err.response?.data?.detail || err.message}`]);
      throw err;
    }
  }, [sessionId]);

  /**
   * 开始轮询任务状态
   */
  const startPolling = useCallback((taskId) => {
    // 清除之前的轮询
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    // 重置错误计数
    pollingErrorCountRef.current = 0;

    // 每 2 秒查询一次状态
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const statusResponse = await api.getTaskStatus(taskId);

        // 成功获取状态，重置错误计数
        pollingErrorCountRef.current = 0;

        setProgress(statusResponse.progress);
        if (statusResponse.current_stage) {
          setLogs((prev) => [...prev, `[${statusResponse.current_stage}] Progress: ${statusResponse.progress}%`]);
        }

        if (statusResponse.status === 'completed') {
          clearInterval(pollingIntervalRef.current);
          setStatus('completed');
          setResult(statusResponse.result);
          setProgress(100);

          // 调试：打印结果数据
          console.log('🔍 Normalization completed. Result:', statusResponse.result);
          console.log('🔍 Has normalized_preview?', !!statusResponse.result?.normalized_preview);

          setLogs((prev) => [
            ...prev,
            `✅ Normalization completed! Time elapsed: ${statusResponse.elapsed_seconds?.toFixed(2)}s`,
          ]);
        } else if (statusResponse.status === 'failed') {
          clearInterval(pollingIntervalRef.current);
          setStatus('error');
          setError(statusResponse.error || 'Processing failed');
          setLogs((prev) => [...prev, `❌ Processing failed: ${statusResponse.error}`]);
        }
      } catch (err) {
        console.error('Polling error:', err);

        // 增加错误计数
        pollingErrorCountRef.current += 1;

        // 如果连续失败5次，停止轮询并报错
        if (pollingErrorCountRef.current >= 5) {
          clearInterval(pollingIntervalRef.current);
          setStatus('error');

          const errorMsg = err.response?.status === 404
            ? 'Task not found (backend may have restarted)'
            : `Connection failed: ${err.message}`;

          setError(errorMsg);
          setLogs((prev) => [...prev, `❌ Polling failed (retried 5 times): ${errorMsg}`]);
        }
      }
    }, 2000);
  }, []);

  /**
   * 重置状态
   */
  const reset = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    setSessionId(null);
    setTaskId(null);
    setStatus('idle');
    setProgress(0);
    setLogs([]);
    setResult(null);
    setError(null);
    setUploadedFileInfo(null);
  }, []);

  /**
   * 清理轮询
   */
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  return {
    // 状态
    sessionId,
    taskId,
    status,
    progress,
    logs,
    result,
    error,
    uploadedFileInfo,

    // 方法
    uploadFile,
    startNormalization,
    reset,
  };
};

export default useNormalizer;

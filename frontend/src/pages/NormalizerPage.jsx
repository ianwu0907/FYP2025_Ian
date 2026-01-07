/**
 * 标准化器主页面
 * 整合所有组件，实现完整的用户界面
 */

import React, { useState } from 'react';
import { Layout, Row, Col, Card, Button, message, Descriptions, Tag, Spin, Divider } from 'antd';
import { RocketOutlined, ReloadOutlined, CheckCircleOutlined, DownloadOutlined, TranslationOutlined } from '@ant-design/icons';
import FileUpload from '../components/FileUpload/FileUpload';
import ProgressDisplay from '../components/ProgressDisplay/ProgressDisplay';
import TableComparison from '../components/TableComparison/TableComparison';
import { useNormalizer } from '../hooks/useNormalizer';
import { api } from '../services/api';
import { useLanguage } from '../contexts/LanguageContext';

const { Header, Content } = Layout;

const NormalizerPage = () => {
  const {
    sessionId,
    taskId,
    status,
    progress,
    logs,
    result,
    error,
    uploadedFileInfo,
    uploadFile,
    startNormalization,
    reset,
  } = useNormalizer();

  const { t, toggleLanguage, currentLanguage } = useLanguage();
  const [configOverrides, setConfigOverrides] = useState({});

  const handleFileUpload = async (file) => {
    try {
      await uploadFile(file);
      message.success(`${file.name} ${t.upload.uploadSuccess}`);
    } catch (err) {
      message.error(`${t.upload.uploadFailed}: ${err.message}`);
    }
  };

  const handleStartNormalization = async () => {
    try {
      await startNormalization(configOverrides);
      message.success(t.normalization.startSuccess);
    } catch (err) {
      message.error(`${t.normalization.startFailed}: ${err.message}`);
    }
  };

  const handleReset = () => {
    reset();
    message.info(t.message.resetSuccess);
  };

  const handleDownload = async () => {
    try {
      message.loading({ content: t.result.downloading, key: 'download' });

      // 调用 API 下载文件
      const blob = await api.downloadResult(taskId);

      // 创建下载链接
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;

      // 从 result 中获取文件名
      const filename = result?.output_path?.split(/[\\/]/).pop() || 'normalized_output.csv';
      link.download = filename;

      // 触发下载
      document.body.appendChild(link);
      link.click();

      // 清理
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      message.success({ content: t.result.downloadSuccess, key: 'download' });
    } catch (err) {
      console.error('Download error:', err);
      message.error({ content: `${t.result.downloadFailed}: ${err.message}`, key: 'download' });
    }
  };

  const getStatusTag = () => {
    switch (status) {
      case 'uploading':
        return <Tag color="blue">{t.status.uploading}</Tag>;
      case 'uploaded':
        return <Tag color="green">{t.status.uploaded}</Tag>;
      case 'processing':
        return <Tag color="orange">{t.status.processing}</Tag>;
      case 'completed':
        return <Tag color="success">{t.status.completed}</Tag>;
      case 'error':
        return <Tag color="error">{t.status.error}</Tag>;
      default:
        return <Tag>{t.status.idle}</Tag>;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <Header
        style={{
          background: '#fff',
          padding: '0 50px',
          borderBottom: '1px solid #e8e8e8',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <h1 style={{ margin: 0, fontSize: 24 }}>
            {t.header.title}
          </h1>
          <div style={{ display: 'flex', gap: '10px' }}>
            <Button
              icon={<TranslationOutlined />}
              onClick={toggleLanguage}
              title={currentLanguage === 'zh' ? 'Switch to English' : '切换到中文'}
            >
              {currentLanguage === 'zh' ? 'EN' : '中文'}
            </Button>
            {status !== 'idle' && (
              <Button icon={<ReloadOutlined />} onClick={handleReset}>
                {t.header.restart}
              </Button>
            )}
          </div>
        </div>
      </Header>

      <Content style={{ padding: '50px' }}>
        <Row gutter={[16, 16]}>
          {/* 步骤 1: 文件上传 */}
          <Col span={24}>
            <Card
              title={<span style={{ fontSize: 18 }}>{t.steps.step1}</span>}
              extra={getStatusTag()}
            >
              <FileUpload
                onUpload={handleFileUpload}
                disabled={status === 'processing' || status === 'completed'}
                loading={status === 'uploading'}
              />

              {uploadedFileInfo && (
                <div style={{ marginTop: 20 }}>
                  <Descriptions title={t.upload.fileInfo} bordered size="small" column={2}>
                    <Descriptions.Item label={t.upload.filename}>
                      {uploadedFileInfo.filename}
                    </Descriptions.Item>
                    <Descriptions.Item label={t.upload.fileType}>
                      {uploadedFileInfo.file_type.toUpperCase()}
                    </Descriptions.Item>
                    <Descriptions.Item label={t.upload.fileSize}>
                      {(uploadedFileInfo.file_size / 1024).toFixed(2)} KB
                    </Descriptions.Item>
                    <Descriptions.Item label={t.upload.dimensions}>
                      {uploadedFileInfo.preview.shape[0]} {t.upload.rows} × {uploadedFileInfo.preview.shape[1]} {t.upload.columns}
                    </Descriptions.Item>
                    <Descriptions.Item label={t.upload.sessionId} span={2}>
                      <code>{sessionId}</code>
                    </Descriptions.Item>
                  </Descriptions>
                </div>
              )}
            </Card>
          </Col>

          {/* 步骤 2: 开始处理 */}
          {status === 'uploaded' && (
            <Col span={24}>
              <Card title={<span style={{ fontSize: 18 }}>{t.steps.step2}</span>}>
                <Button
                  type="primary"
                  size="large"
                  icon={<RocketOutlined />}
                  onClick={handleStartNormalization}
                  style={{ width: '100%', height: 50, fontSize: 16 }}
                >
                  {t.normalization.start}
                </Button>
              </Card>
            </Col>
          )}

          {/* 步骤 3: 处理进度 */}
          {(status === 'processing' || status === 'completed') && (
            <Col span={24}>
              <Card
                title={<span style={{ fontSize: 18 }}>{t.steps.step3}</span>}
                extra={status === 'completed' && <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 24 }} />}
              >
                <ProgressDisplay progress={progress} logs={logs} status={status} />
              </Card>
            </Col>
          )}

          {/* 步骤 4: 处理结果 */}
          {status === 'completed' && result && (
            <Col span={24}>
              <Card title={<span style={{ fontSize: 18 }}>{t.steps.step4}</span>}>
                <Descriptions bordered column={2}>
                  <Descriptions.Item label={t.result.outputPath} span={2}>
                    <code>{result.output_path}</code>
                  </Descriptions.Item>
                  <Descriptions.Item label={t.result.tableCount}>
                    {result.num_tables}
                  </Descriptions.Item>
                  <Descriptions.Item label={t.result.detectionMethod}>
                    {result.detection_method}
                  </Descriptions.Item>
                  <Descriptions.Item label={t.result.processingTime} span={2}>
                    {result.elapsed_seconds?.toFixed(2)} {t.result.seconds}
                  </Descriptions.Item>
                </Descriptions>

                <div style={{ marginTop: 20, textAlign: 'center' }}>
                  <Button
                    type="primary"
                    size="large"
                    icon={<DownloadOutlined />}
                    onClick={handleDownload}
                  >
                    {t.result.downloadResult}
                  </Button>
                </div>

                {/* 表格对比视图 */}
                {(() => {
                  // 调试日志 (Debug logs)
                  console.log('🔍 Rendering comparison check:');
                  console.log('  - result:', result);
                  console.log('  - result.normalized_preview:', result?.normalized_preview);
                  console.log('  - uploadedFileInfo:', uploadedFileInfo);
                  console.log('  - uploadedFileInfo.preview:', uploadedFileInfo?.preview);

                  return result?.normalized_preview && uploadedFileInfo?.preview ? (
                    <>
                      <Divider orientation="left" style={{ fontSize: 16, fontWeight: 'bold' }}>
                        {t.result.tableComparison}
                      </Divider>
                      <TableComparison
                        originalData={uploadedFileInfo.preview}
                        normalizedData={result.normalized_preview}
                      />
                    </>
                  ) : (
                    <div style={{ marginTop: 20, padding: 20, background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 4 }}>
                      <p style={{ margin: 0, color: '#856404' }}>
                        ⚠️ {currentLanguage === 'zh' ? '调试信息：' : 'Debug Info: '}{t.result.comparisonWarning}
                        <br />
                        - {t.result.normalizedPreviewExists}: {result?.normalized_preview ? '✅' : '❌'}
                        <br />
                        - {t.result.originalPreviewExists}: {uploadedFileInfo?.preview ? '✅' : '❌'}
                      </p>
                    </div>
                  );
                })()}
              </Card>
            </Col>
          )}

          {/* 错误显示 */}
          {status === 'error' && error && (
            <Col span={24}>
              <Card title={t.error.title} style={{ borderColor: '#ff4d4f' }}>
                <p style={{ color: '#ff4d4f', fontSize: 16 }}>
                  {error}
                </p>
                <Button onClick={handleReset}>{t.error.retry}</Button>
              </Card>
            </Col>
          )}
        </Row>
      </Content>
    </Layout>
  );
};

export default NormalizerPage;

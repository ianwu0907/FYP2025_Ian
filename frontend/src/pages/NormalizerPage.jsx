/**
 * 标准化器主页面
 * 整合所有组件，实现完整的用户界面
 */

import React, { useState } from 'react';
import { Layout, Row, Col, Card, Button, message, Descriptions, Tag, Spin, Divider, Tooltip, Grid } from 'antd';

const { useBreakpoint } = Grid;
import { RocketOutlined, ReloadOutlined, CheckCircleOutlined, DownloadOutlined, TranslationOutlined, FileExcelOutlined } from '@ant-design/icons';
import FileUpload from '../components/FileUpload/FileUpload';
import ProgressDisplay from '../components/ProgressDisplay/ProgressDisplay';
import TableComparison from '../components/TableComparison/TableComparison';
import IrregularityPanel from '../components/IrregularityPanel/IrregularityPanel';
import MetricsPanel from '../components/MetricsPanel/MetricsPanel';
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
    currentStage,
    uploadFile,
    startNormalization,
    reset,
    retry,
  } = useNormalizer();

  const { t, toggleLanguage, currentLanguage } = useLanguage();
  const [configOverrides, setConfigOverrides] = useState({});
  const [demoLoading, setDemoLoading] = useState(null);
  const screens = useBreakpoint();
  const isMobile = !screens.md;

  const handleDemoFile = async (demoFile) => {
    setDemoLoading(demoFile);
    try {
      const res = await fetch(`/demos/${demoFile}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const file = new File([blob], demoFile, { type: blob.type || 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      await uploadFile(file);
      message.success(`${demoFile} ${t.upload.uploadSuccess}`);
    } catch (err) {
      message.error(`${t.upload.uploadFailed}: ${err.message}`);
    } finally {
      setDemoLoading(null);
    }
  };

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
    const tagStyle = {
      fontSize: '14px',
      fontWeight: 'bold',
      padding: '4px 12px',
      borderRadius: '6px',
    };

    switch (status) {
      case 'uploading':
        return <Tag color="blue" style={tagStyle}>{t.status.uploading}</Tag>;
      case 'uploaded':
        return <Tag color="success" style={tagStyle}>{t.status.uploaded}</Tag>;
      case 'processing':
        return <Tag color="processing" style={tagStyle}>{t.status.processing}</Tag>;
      case 'completed':
        return <Tag color="success" style={tagStyle}>{t.status.completed}</Tag>;
      case 'error':
        return <Tag color="error" style={tagStyle}>{t.status.error}</Tag>;
      default:
        return <Tag color="default" style={tagStyle}>{t.status.idle}</Tag>;
    }
  };

  return (
    <Layout style={{ minHeight: '100vh', background: '#9e9cf1ff' }}>
      <Header
        style={{
          background: 'linear-gradient(90deg, #96aaebff 0%, #9eabebff 100%)',
          padding: isMobile ? '0 16px' : '0 50px',
          height: isMobile ? 'auto' : 64,
          lineHeight: isMobile ? '1' : '64px',
          borderBottom: 'none',
          boxShadow: '0 4px 12px rgba(102, 126, 234, 0.3)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: isMobile ? '12px 0' : 0 }}>
          <h1 style={{ margin: 0, fontSize: isMobile ? 16 : 24, color: '#fff', fontWeight: 'bold' }}>
            {t.header.title}
          </h1>
          <div style={{ display: 'flex', gap: '10px' }}>
            <Button
              icon={<TranslationOutlined />}
              onClick={toggleLanguage}
              title={currentLanguage === 'zh' ? 'Switch to English' : '切换到中文'}
              style={{
                background: 'rgba(255, 255, 255, 0.2)',
                border: '1px solid rgba(255, 255, 255, 0.3)',
                color: '#fff',
                fontWeight: 'bold',
              }}
            >
              {currentLanguage === 'zh' ? 'EN' : '中文'}
            </Button>
            {status !== 'idle' && (
              <Button
                icon={<ReloadOutlined />}
                onClick={handleReset}
                style={{
                  background: 'rgba(255, 255, 255, 0.2)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  color: '#fff',
                  fontWeight: 'bold',
                }}
              >
                {t.header.restart}
              </Button>
            )}
          </div>
        </div>
      </Header>

      <Content style={{ padding: isMobile ? '16px' : '50px' }}>
        <Row gutter={[16, 16]}>
          {/* 步骤 1: 文件上传 */}
          <Col span={24}>
            <Card
              title={<span style={{ fontSize: 18, fontWeight: 'bold', color: '#667eea' }}>{t.steps.step1}</span>}
              extra={getStatusTag()}
              style={{
                borderRadius: '12px',
                boxShadow: '0 4px 20px rgba(102, 126, 234, 0.15)',
                border: 'none',
              }}
              headStyle={{
                background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                borderRadius: '12px 12px 0 0',
              }}
            >
              <FileUpload
                onUpload={handleFileUpload}
                disabled={status === 'processing' || status === 'completed'}
                loading={status === 'uploading'}
              />

              {/* Demo Files */}
              {status === 'idle' && (
                <div style={{ marginTop: 16 }}>
                  <Divider plain style={{ color: '#999', fontSize: 13 }}>{t.demo.title}</Divider>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
                    {t.demo.files.map((demo) => (
                      <Tooltip key={demo.file} title={demo.desc}>
                        <Button
                          icon={<FileExcelOutlined style={{ color: '#217346' }} />}
                          loading={demoLoading === demo.file}
                          onClick={() => handleDemoFile(demo.file)}
                          style={{
                            borderRadius: 8,
                            borderColor: '#217346',
                            color: '#217346',
                            fontWeight: 500,
                          }}
                        >
                          {demo.name}
                        </Button>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              )}

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
                    {/* <Descriptions.Item label={t.upload.sessionId} span={2}>
                      <code>{sessionId}</code>
                    </Descriptions.Item> */}
                  </Descriptions>
                </div>
              )}
            </Card>
          </Col>

          {/* 步骤 2: 开始处理 */}
          {status === 'uploaded' && (
            <Col span={24}>
              <Card
                title={<span style={{ fontSize: 18, fontWeight: 'bold', color: '#667eea' }}>{t.steps.step2}</span>}
                style={{
                  borderRadius: '12px',
                  boxShadow: '0 4px 20px rgba(102, 126, 234, 0.15)',
                  border: 'none',
                }}
                headStyle={{
                  background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                  borderRadius: '12px 12px 0 0',
                }}
              >
                <Button
                  type="primary"
                  size="large"
                  icon={<RocketOutlined />}
                  onClick={handleStartNormalization}
                  style={{
                    width: '100%',
                    height: 60,
                    fontSize: 18,
                    fontWeight: 'bold',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    border: 'none',
                    borderRadius: '8px',
                    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
                  }}
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
                title={<span style={{ fontSize: 18, fontWeight: 'bold', color: '#667eea' }}>{t.steps.step3}</span>}
                extra={status === 'completed' && <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 24 }} />}
                style={{
                  borderRadius: '12px',
                  boxShadow: '0 4px 20px rgba(102, 126, 234, 0.15)',
                  border: 'none',
                }}
                headStyle={{
                  background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                  borderRadius: '12px 12px 0 0',
                }}
              >
                <ProgressDisplay progress={progress} logs={logs} status={status} currentStage={currentStage} />
              </Card>
            </Col>
          )}

          {/* 步骤 4: 处理结果 */}
          {status === 'completed' && result && (
            <Col span={24}>
              <Card
                title={<span style={{ fontSize: 18, fontWeight: 'bold', color: '#667eea' }}>{t.steps.step4}</span>}
                style={{
                  borderRadius: '12px',
                  boxShadow: '0 4px 20px rgba(102, 126, 234, 0.15)',
                  border: 'none',
                }}
                headStyle={{
                  background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                  borderRadius: '12px 12px 0 0',
                }}
              >
                {/* <Descriptions bordered column={2}>
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
                </Descriptions> */}



                {/* Irregularities + Metrics side by side */}
                <Row gutter={[24, 16]} style={{ marginBottom: 8 }}>
                  <Col xs={24} lg={10}>
                    <Divider orientation="left" style={{ fontSize: 15, fontWeight: 'bold', margin: '12px 0 10px' }}>
                      {currentLanguage === 'zh' ? '检测到的结构异常' : 'Detected Irregularities'}
                    </Divider>
                    <IrregularityPanel
                      irregularities={result?.irregularities || []}
                      labels={result?.irregularity_labels || []}
                    />
                  </Col>
                  <Col xs={24} lg={14}>
                    <Divider orientation="left" style={{ fontSize: 15, fontWeight: 'bold', margin: '12px 0 10px' }}>
                      {currentLanguage === 'zh' ? '整洁度指标 (前后对比)' : 'Tidiness Metrics (Before vs After)'}
                    </Divider>
                    <MetricsPanel
                      metricsBefore={result?.metrics_before || {}}
                      metricsAfter={result?.metrics_after || {}}
                      metricsComparison={result?.metrics_comparison || {}}
                    />
                  </Col>
                </Row>

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
                <div style={{ marginTop: 20, textAlign: 'center' }}>
                  <Button
                    type="primary"
                    size="large"
                    icon={<DownloadOutlined />}
                    onClick={handleDownload}
                    style={{
                      height: 50,
                      fontSize: 16,
                      fontWeight: 'bold',
                      paddingLeft: 40,
                      paddingRight: 40,
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      border: 'none',
                      borderRadius: '8px',
                      boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
                    }}
                  >
                    {t.result.downloadResult}
                  </Button>
                </div>
              </Card>
            </Col>
          )}

          {/* 错误显示 */}
          {status === 'error' && error && (
            <Col span={24}>
              <Card
                title={
                  <span style={{ fontSize: 18, fontWeight: 'bold', color: '#ff4d4f' }}>
                    ❌ {currentLanguage === 'zh' ? '标准化失败' : 'Normalization Failed'}
                  </span>
                }
                style={{
                  borderRadius: '12px',
                  boxShadow: '0 4px 20px rgba(255, 77, 79, 0.15)',
                  border: '2px solid #ff4d4f',
                }}
                headStyle={{
                  background: 'linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%)',
                  borderRadius: '12px 12px 0 0',
                }}
              >
                {/* Error message */}
                <div style={{
                  background: '#fff2f0',
                  border: '1px solid #ffccc7',
                  borderRadius: 8,
                  padding: '12px 16px',
                  marginBottom: 20,
                  fontFamily: 'monospace',
                  fontSize: 13,
                  color: '#cf1322',
                }}>
                  {error}
                </div>

                {/* Action buttons */}
                <div style={{ display: 'flex', gap: 12, marginBottom: uploadedFileInfo?.preview ? 24 : 0 }}>
                  <Button
                    type="primary"
                    icon={<RocketOutlined />}
                    onClick={() => { retry(); }}
                    style={{
                      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      border: 'none',
                      fontWeight: 'bold',
                      borderRadius: '6px',
                    }}
                  >
                    {currentLanguage === 'zh' ? '重新尝试' : 'Retry Normalization'}
                  </Button>
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={handleReset}
                    style={{ borderRadius: '6px' }}
                  >
                    {currentLanguage === 'zh' ? '上传新文件' : 'Upload New File'}
                  </Button>
                </div>

                {/* Original table preview */}
                {uploadedFileInfo?.preview && (
                  <>
                    <Divider orientation="left" style={{ fontSize: 15, fontWeight: 'bold', color: '#595959' }}>
                      {currentLanguage === 'zh' ? '原始表格（未做处理）' : 'Original Table (unmodified)'}
                    </Divider>
                    <TableComparison
                      originalData={uploadedFileInfo.preview}
                      normalizedData={null}
                      singleMode
                    />
                  </>
                )}
              </Card>
            </Col>
          )}
        </Row>
      </Content>
    </Layout>
  );
};

export default NormalizerPage;

/**
 * 标准化器主页面
 * 整合所有组件，实现完整的用户界面
 */

import React, { useState } from 'react';
import { Layout, Row, Col, Card, Button, message, Descriptions, Tag, Spin, Divider } from 'antd';
import { RocketOutlined, ReloadOutlined, CheckCircleOutlined, DownloadOutlined } from '@ant-design/icons';
import FileUpload from '../components/FileUpload/FileUpload';
import ProgressDisplay from '../components/ProgressDisplay/ProgressDisplay';
import TableComparison from '../components/TableComparison/TableComparison';
import { useNormalizer } from '../hooks/useNormalizer';
import { api } from '../services/api';

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

  const [configOverrides, setConfigOverrides] = useState({});

  const handleFileUpload = async (file) => {
    try {
      await uploadFile(file);
      message.success(`文件 ${file.name} 上传成功！`);
    } catch (err) {
      message.error(`上传失败: ${err.message}`);
    }
  };

  const handleStartNormalization = async () => {
    try {
      await startNormalization(configOverrides);
      message.success('标准化处理已开始！');
    } catch (err) {
      message.error(`启动失败: ${err.message}`);
    }
  };

  const handleReset = () => {
    reset();
    message.info('已重置，可以上传新文件');
  };

  const handleDownload = async () => {
    try {
      message.loading({ content: '正在准备下载...', key: 'download' });

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

      message.success({ content: '下载成功！', key: 'download' });
    } catch (err) {
      console.error('Download error:', err);
      message.error({ content: `下载失败: ${err.message}`, key: 'download' });
    }
  };

  const getStatusTag = () => {
    switch (status) {
      case 'uploading':
        return <Tag color="blue">正在上传...</Tag>;
      case 'uploaded':
        return <Tag color="green">上传完成</Tag>;
      case 'processing':
        return <Tag color="orange">处理中...</Tag>;
      case 'completed':
        return <Tag color="success">处理完成</Tag>;
      case 'error':
        return <Tag color="error">错误</Tag>;
      default:
        return <Tag>等待上传</Tag>;
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
            📊 Spreadsheet Normalizer
          </h1>
          {status !== 'idle' && (
            <Button icon={<ReloadOutlined />} onClick={handleReset}>
              重新开始
            </Button>
          )}
        </div>
      </Header>

      <Content style={{ padding: '50px' }}>
        <Row gutter={[16, 16]}>
          {/* 步骤 1: 文件上传 */}
          <Col span={24}>
            <Card
              title={<span style={{ fontSize: 18 }}>步骤 1: 上传文件</span>}
              extra={getStatusTag()}
            >
              <FileUpload
                onUpload={handleFileUpload}
                disabled={status === 'processing' || status === 'completed'}
                loading={status === 'uploading'}
              />

              {uploadedFileInfo && (
                <div style={{ marginTop: 20 }}>
                  <Descriptions title="文件信息" bordered size="small" column={2}>
                    <Descriptions.Item label="文件名">
                      {uploadedFileInfo.filename}
                    </Descriptions.Item>
                    <Descriptions.Item label="文件类型">
                      {uploadedFileInfo.file_type.toUpperCase()}
                    </Descriptions.Item>
                    <Descriptions.Item label="文件大小">
                      {(uploadedFileInfo.file_size / 1024).toFixed(2)} KB
                    </Descriptions.Item>
                    <Descriptions.Item label="数据维度">
                      {uploadedFileInfo.preview.shape[0]} 行 × {uploadedFileInfo.preview.shape[1]} 列
                    </Descriptions.Item>
                    <Descriptions.Item label="Session ID" span={2}>
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
              <Card title={<span style={{ fontSize: 18 }}>步骤 2: 开始标准化</span>}>
                <Button
                  type="primary"
                  size="large"
                  icon={<RocketOutlined />}
                  onClick={handleStartNormalization}
                  style={{ width: '100%', height: 50, fontSize: 16 }}
                >
                  开始标准化处理
                </Button>
              </Card>
            </Col>
          )}

          {/* 步骤 3: 处理进度 */}
          {(status === 'processing' || status === 'completed') && (
            <Col span={24}>
              <Card
                title={<span style={{ fontSize: 18 }}>步骤 3: 处理进度</span>}
                extra={status === 'completed' && <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 24 }} />}
              >
                <ProgressDisplay progress={progress} logs={logs} status={status} />
              </Card>
            </Col>
          )}

          {/* 步骤 4: 处理结果 */}
          {status === 'completed' && result && (
            <Col span={24}>
              <Card title={<span style={{ fontSize: 18 }}>步骤 4: 处理结果</span>}>
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="输出路径" span={2}>
                    <code>{result.output_path}</code>
                  </Descriptions.Item>
                  <Descriptions.Item label="表格数量">
                    {result.num_tables}
                  </Descriptions.Item>
                  <Descriptions.Item label="检测方法">
                    {result.detection_method}
                  </Descriptions.Item>
                  <Descriptions.Item label="处理时间" span={2}>
                    {result.elapsed_seconds?.toFixed(2)} 秒
                  </Descriptions.Item>
                </Descriptions>

                <div style={{ marginTop: 20, textAlign: 'center' }}>
                  <Button
                    type="primary"
                    size="large"
                    icon={<DownloadOutlined />}
                    onClick={handleDownload}
                  >
                    下载标准化结果
                  </Button>
                </div>

                {/* 表格对比视图 */}
                {(() => {
                  // 调试日志
                  console.log('🔍 Rendering comparison check:');
                  console.log('  - result:', result);
                  console.log('  - result.normalized_preview:', result?.normalized_preview);
                  console.log('  - uploadedFileInfo:', uploadedFileInfo);
                  console.log('  - uploadedFileInfo.preview:', uploadedFileInfo?.preview);

                  return result?.normalized_preview && uploadedFileInfo?.preview ? (
                    <>
                      <Divider orientation="left" style={{ fontSize: 16, fontWeight: 'bold' }}>
                        表格对比
                      </Divider>
                      <TableComparison
                        originalData={uploadedFileInfo.preview}
                        normalizedData={result.normalized_preview}
                      />
                    </>
                  ) : (
                    <div style={{ marginTop: 20, padding: 20, background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 4 }}>
                      <p style={{ margin: 0, color: '#856404' }}>
                        ⚠️ 调试信息：无法显示对比视图
                        <br />
                        - normalized_preview 存在: {result?.normalized_preview ? '✅' : '❌'}
                        <br />
                        - uploadedFileInfo.preview 存在: {uploadedFileInfo?.preview ? '✅' : '❌'}
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
              <Card title="错误信息" style={{ borderColor: '#ff4d4f' }}>
                <p style={{ color: '#ff4d4f', fontSize: 16 }}>
                  {error}
                </p>
                <Button onClick={handleReset}>重新尝试</Button>
              </Card>
            </Col>
          )}
        </Row>
      </Content>
    </Layout>
  );
};

export default NormalizerPage;

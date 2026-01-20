/**
 * 进度显示组件
 * 显示处理进度和日志
 */

import React from 'react';
import { Progress, Card, List } from 'antd';
import { useLanguage } from '../../contexts/LanguageContext';

const ProgressDisplay = ({ progress, logs, status }) => {
  const { t } = useLanguage();
  const getProgressStatus = () => {
    if (status === 'error') return 'exception';
    if (status === 'completed') return 'success';
    return 'active';
  };

  return (
    <div style={{ marginTop: 20 }}>
      <Progress
        percent={progress}
        status={getProgressStatus()}
        strokeColor={{
          '0%': '#667eea',
          '100%': '#764ba2',
        }}
        strokeWidth={12}
        style={{
          fontSize: 16,
        }}
      />

      <Card
        title={t.progress.processingLogs}
        size="small"
        style={{
          marginTop: 16,
          maxHeight: 300,
          overflow: 'auto',
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(102, 126, 234, 0.1)',
        }}
        headStyle={{
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          fontWeight: 'bold',
          color: '#667eea',
        }}
      >
        <List
          size="small"
          dataSource={logs}
          renderItem={(log, index) => (
            <List.Item key={index}>
              <span style={{ fontFamily: 'monospace', fontSize: 12 }}>
                {log}
              </span>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default ProgressDisplay;

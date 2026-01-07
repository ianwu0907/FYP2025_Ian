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
          '0%': '#108ee9',
          '100%': '#87d068',
        }}
      />

      <Card
        title={t.progress.processingLogs}
        size="small"
        style={{
          marginTop: 16,
          maxHeight: 300,
          overflow: 'auto',
        }}
      >
        <List
          size="small"
          dataSource={logs}
          renderItem={(log, index) => (
            <List.Item key={index}>
              <span style={{ fontFamily: 'monospace', fontSize: 12 }}>
                [{new Date().toLocaleTimeString()}] {log}
              </span>
            </List.Item>
          )}
        />
      </Card>
    </div>
  );
};

export default ProgressDisplay;

/**
 * 进度显示组件
 * 显示 pipeline 各阶段状态 + 进度条 + 日志
 */

import React, { useEffect, useRef } from 'react';
import { Progress, Steps, Card, Grid } from 'antd';

const { useBreakpoint } = Grid;
import {
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import { useLanguage } from '../../contexts/LanguageContext';

const STAGES = [
  { key: 'encoding',              label: 'Spreadsheet Encoding',    labelZh: '电子表格编码' },
  { key: 'irregularity_detection', label: 'Irregularity Detection',  labelZh: '不规则检测' },
  { key: 'schema_estimation',     label: 'Schema Estimation',        labelZh: '模式估计' },
  { key: 'transformation',        label: 'Transformation',           labelZh: '转换生成' },
];

const STAGE_ORDER = ['initializing', 'encoding', 'irregularity_detection', 'schema_estimation', 'transformation', 'completed'];

function getStepStatus(stageKey, currentStage, overallStatus) {
  if (overallStatus === 'error') {
    const currentIdx = STAGE_ORDER.indexOf(currentStage);
    const stageIdx   = STAGE_ORDER.indexOf(stageKey);
    if (stageIdx < currentIdx) return 'finish';
    if (stageIdx === currentIdx) return 'error';
    return 'wait';
  }
  if (overallStatus === 'completed') return 'finish';
  const currentIdx = STAGE_ORDER.indexOf(currentStage);
  const stageIdx   = STAGE_ORDER.indexOf(stageKey);
  if (currentStage === 'completed' || stageIdx < currentIdx) return 'finish';
  if (stageIdx === currentIdx) return 'process';
  return 'wait';
}

function getStepIcon(stepStatus) {
  switch (stepStatus) {
    case 'finish':  return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    case 'process': return <LoadingOutlined style={{ color: '#667eea' }} />;
    case 'error':   return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
    default:        return <ClockCircleOutlined style={{ color: '#bbb' }} />;
  }
}

const ProgressDisplay = ({ progress, logs, status, currentStage }) => {
  const { t, currentLanguage } = useLanguage();
  const logEndRef = useRef(null);
  const screens = useBreakpoint();

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getProgressStatus = () => {
    if (status === 'error') return 'exception';
    if (status === 'completed') return 'success';
    return 'active';
  };

  const stepsItems = STAGES.map((s) => {
    const stepStatus = getStepStatus(s.key, currentStage, status === 'error' ? 'error' : status);
    return {
      key: s.key,
      title: currentLanguage === 'zh' ? s.labelZh : s.label,
      icon: getStepIcon(stepStatus),
      status: stepStatus,
    };
  });

  // Determine current step index for Steps component
  const currentStepIdx = Math.max(0, STAGES.findIndex(s => s.key === currentStage));

  return (
    <div style={{ marginTop: 20 }}>
      {/* Pipeline Steps */}
      <Steps
        items={stepsItems}
        current={currentStepIdx}
        direction={screens.md ? 'horizontal' : 'vertical'}
        style={{ marginBottom: 24 }}
        size="small"
      />

      {/* Progress Bar */}
      <Progress
        percent={progress}
        status={getProgressStatus()}
        strokeColor={{ '0%': '#667eea', '100%': '#764ba2' }}
        strokeWidth={10}
      />

      {/* Log Panel */}
      <Card
        title={t.progress.processingLogs}
        size="small"
        style={{
          marginTop: 16,
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(102, 126, 234, 0.1)',
        }}
        headStyle={{
          background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
          fontWeight: 'bold',
          color: '#667eea',
        }}
        bodyStyle={{ padding: 0 }}
      >
        <div
          style={{
            maxHeight: 220,
            overflowY: 'auto',
            padding: '8px 12px',
            background: '#1e1e2e',
            borderRadius: '0 0 8px 8px',
          }}
        >
          {logs.map((log, i) => (
            <div key={i} style={{ fontFamily: 'monospace', fontSize: 12, color: '#cdd6f4', lineHeight: '1.6' }}>
              {log}
            </div>
          ))}
          <div ref={logEndRef} />
        </div>
      </Card>
    </div>
  );
};

export default ProgressDisplay;

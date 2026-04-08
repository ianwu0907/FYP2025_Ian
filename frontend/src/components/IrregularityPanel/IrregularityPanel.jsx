/**
 * IrregularityPanel
 * Displays detected spreadsheet irregularities with evidence after pipeline completes.
 */

import React, { useState } from 'react';
import { Collapse, Tag, Empty, Typography, Badge } from 'antd';
import {
  WarningOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { useLanguage } from '../../contexts/LanguageContext';

const { Text, Paragraph } = Typography;

// Color map for irregularity labels
const LABEL_COLORS = {
  WIDE_FORMAT:                 'blue',
  MULTI_LEVEL_HEADER:          'purple',
  MERGED_HEADER:               'purple',
  SPARSE_HEADER:               'geekblue',
  HIERARCHICAL_ROW_LABELS:     'cyan',
  AGGREGATE_ROWS:              'orange',
  METADATA_ROWS:               'gold',
  BILINGUAL_ALTERNATING_ROWS:  'volcano',
  INLINE_BILINGUAL:            'volcano',
  MIXED_TYPE_COLUMN:           'red',
  IMPLICIT_MISSING:            'magenta',
  TRANSPOSED_TABLE:            'lime',
};

function labelColor(label) {
  return LABEL_COLORS[label] || 'default';
}

const IrregularityPanel = ({ irregularities = [], labels = [] }) => {
  const { t, currentLanguage } = useLanguage();
  const [activeKeys, setActiveKeys] = useState([]);

  const isZh = currentLanguage === 'zh';

  if (!irregularities || irregularities.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '24px 0' }}>
        <CheckCircleOutlined style={{ fontSize: 32, color: '#52c41a', marginBottom: 8 }} />
        <div style={{ color: '#52c41a', fontWeight: 600, fontSize: 15 }}>
          {isZh ? '未检测到结构异常' : 'No structural irregularities detected'}
        </div>
      </div>
    );
  }

  const items = irregularities.map((ir, idx) => ({
    key: String(idx),
    label: (
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <WarningOutlined style={{ color: '#faad14' }} />
        <Tag color={labelColor(ir.label)} style={{ fontWeight: 600, fontFamily: 'monospace', fontSize: 12 }}>
          {ir.label}
        </Tag>
        {ir.details && (
          <Text type="secondary" style={{ fontSize: 13 }}>
            {ir.details}
          </Text>
        )}
      </div>
    ),
    children: (
      <div style={{ padding: '4px 0' }}>
        {ir.evidence && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
              <InfoCircleOutlined style={{ color: '#1890ff' }} />
              <Text strong style={{ fontSize: 13 }}>
                {isZh ? '证据' : 'Evidence'}
              </Text>
            </div>
            <Paragraph
              style={{
                background: '#f6f8fa',
                border: '1px solid #e1e4e8',
                borderRadius: 6,
                padding: '10px 14px',
                fontFamily: 'monospace',
                fontSize: 13,
                margin: 0,
                whiteSpace: 'pre-wrap',
                color: '#24292e',
              }}
            >
              {ir.evidence}
            </Paragraph>
          </div>
        )}
      </div>
    ),
  }));

  return (
    <div>
      {/* Summary badges */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
        <Text type="secondary" style={{ fontSize: 13, alignSelf: 'center' }}>
          {isZh ? `检测到 ${irregularities.length} 个异常：` : `${irregularities.length} irregularit${irregularities.length > 1 ? 'ies' : 'y'} detected:`}
        </Text>
        {labels.map(label => (
          <Tag key={label} color={labelColor(label)} style={{ fontFamily: 'monospace', fontSize: 11 }}>
            {label}
          </Tag>
        ))}
      </div>

      {/* Collapsible detail cards */}
      <Collapse
        items={items}
        activeKey={activeKeys}
        onChange={setActiveKeys}
        style={{ borderRadius: 8 }}
        size="small"
      />
    </div>
  );
};

export default IrregularityPanel;

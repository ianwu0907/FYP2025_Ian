/**
 * MetricsPanel
 * Shows before/after tidiness metrics comparison after pipeline completes.
 */

import React from 'react';
import { Table, Tag, Tooltip, Typography, Row, Col, Statistic } from 'antd';
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  MinusOutlined,
} from '@ant-design/icons';
import { useLanguage } from '../../contexts/LanguageContext';

const { Text } = Typography;

// Human-readable names + descriptions
const METRIC_META = {
  cell_coverage:                { label: 'Cell Coverage',               labelZh: '单元格覆盖率',     desc: 'Fraction of non-empty cells',                         higherBetter: true  },
  row_completeness_uniformity:  { label: 'Row Uniformity',              labelZh: '行完整性均匀度',    desc: 'How uniformly rows are filled (low std of fill rate)', higherBetter: true  },
  column_type_homogeneity:      { label: 'Column Type Homogeneity',     labelZh: '列类型一致性',     desc: 'Fraction of columns with consistent data types',       higherBetter: true  },
  data_row_ratio:               { label: 'Data Row Ratio',              labelZh: '数据行占比',       desc: 'Fraction of rows that are actual data (not headers)',  higherBetter: true  },
  column_completeness_min:      { label: 'Min Column Completeness',     labelZh: '最低列完整度',     desc: 'Completeness of the least-filled column',             higherBetter: true  },
  header_uniqueness:            { label: 'Header Uniqueness',           labelZh: '表头唯一性',       desc: 'Fraction of unique column headers',                   higherBetter: true  },
  type_consistency:             { label: 'Type Consistency',            labelZh: '类型一致性',       desc: 'Per-column type consistency score',                   higherBetter: true  },
  groupby_queryability:         { label: 'GroupBy Queryability',        labelZh: 'GroupBy可查询性',  desc: 'How well the table supports group-by queries',        higherBetter: true  },
  substring_containment:        { label: 'Substring Containment',       labelZh: '子串包含率',       desc: 'Fraction of values that contain another value (lower = cleaner)', higherBetter: false },
  inter_column_nmi:             { label: 'Inter-Column NMI',            labelZh: '列间归一化互信息',  desc: 'Normalized mutual information between columns (lower = less redundancy)', higherBetter: false },
};

const METRIC_ORDER = Object.keys(METRIC_META);

function fmt(v) {
  if (v === undefined || v === null) return '—';
  return (v * 100).toFixed(1) + '%';
}

function DeltaBadge({ delta, higherBetter, direction }) {
  if (!direction || direction === '= unchanged') {
    return <Tag color="default" icon={<MinusOutlined />} style={{ fontSize: 11 }}>—</Tag>;
  }
  const improved = direction.startsWith('✓');
  const color = improved ? '#52c41a' : '#ff4d4f';
  const Icon  = improved ? ArrowUpOutlined : ArrowDownOutlined;
  const sign  = delta > 0 ? '+' : '';
  return (
    <Tag color={improved ? 'success' : 'error'} icon={<Icon />} style={{ fontSize: 11 }}>
      {sign}{(delta * 100).toFixed(1)}%
    </Tag>
  );
}

const MetricsPanel = ({ metricsBefore = {}, metricsAfter = {}, metricsComparison = {} }) => {
  const { currentLanguage } = useLanguage();
  const isZh = currentLanguage === 'zh';

  // Summary stats
  const improved = METRIC_ORDER.filter(k => metricsComparison[k]?.direction?.startsWith('✓')).length;
  const degraded = METRIC_ORDER.filter(k => metricsComparison[k]?.direction?.startsWith('✗')).length;
  const unchanged = METRIC_ORDER.length - improved - degraded;

  // Shape info
  const shapeBefore = metricsBefore.shape || {};
  const shapeAfter  = metricsAfter.shape  || {};

  const columns = [
    {
      title: isZh ? '指标' : 'Metric',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (_, row) => (
        <Tooltip title={row.desc}>
          <Text style={{ fontSize: 13, cursor: 'help' }}>
            {isZh ? row.labelZh : row.label}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: isZh ? '原始' : 'Before',
      dataIndex: 'before',
      key: 'before',
      align: 'center',
      width: 90,
      render: v => <Text type="secondary" style={{ fontSize: 13 }}>{fmt(v)}</Text>,
    },
    {
      title: isZh ? '处理后' : 'After',
      dataIndex: 'after',
      key: 'after',
      align: 'center',
      width: 90,
      render: (v, row) => {
        const cmp = metricsComparison[row.key];
        const improved = cmp?.direction?.startsWith('✓');
        const degraded = cmp?.direction?.startsWith('✗');
        const color = improved ? '#52c41a' : degraded ? '#ff4d4f' : undefined;
        return <Text strong style={{ fontSize: 13, color }}>{fmt(v)}</Text>;
      },
    },
    {
      title: isZh ? '变化' : 'Delta',
      key: 'delta',
      align: 'center',
      width: 100,
      render: (_, row) => {
        const cmp = metricsComparison[row.key];
        if (!cmp) return '—';
        return <DeltaBadge delta={cmp.delta} higherBetter={row.higherBetter} direction={cmp.direction} />;
      },
    },
  ];

  const dataSource = METRIC_ORDER.map(key => ({
    key,
    ...METRIC_META[key],
    before: metricsBefore[key],
    after:  metricsAfter[key],
  }));

  return (
    <div>
      {/* Summary row */}
      <Row gutter={16} style={{ marginBottom: 20 }}>
        <Col span={6}>
          <div style={{ background: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: 8, padding: '10px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#52c41a' }}>{improved}</div>
            <div style={{ fontSize: 12, color: '#52c41a' }}>{isZh ? '改善' : 'Improved'}</div>
          </div>
        </Col>
        <Col span={6}>
          <div style={{ background: '#fff2f0', border: '1px solid #ffccc7', borderRadius: 8, padding: '10px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#ff4d4f' }}>{degraded}</div>
            <div style={{ fontSize: 12, color: '#ff4d4f' }}>{isZh ? '下降' : 'Degraded'}</div>
          </div>
        </Col>
        <Col span={6}>
          <div style={{ background: '#fafafa', border: '1px solid #d9d9d9', borderRadius: 8, padding: '10px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#888' }}>{unchanged}</div>
            <div style={{ fontSize: 12, color: '#888' }}>{isZh ? '不变' : 'Unchanged'}</div>
          </div>
        </Col>
        <Col span={6}>
          <div style={{ background: '#f0f5ff', border: '1px solid #adc6ff', borderRadius: 8, padding: '10px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: '#2f54eb' }}>
              {shapeBefore.rows ?? '?'} × {shapeBefore.cols ?? '?'}
            </div>
            <div style={{ fontSize: 11, color: '#888', margin: '2px 0' }}>→</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#2f54eb' }}>
              {shapeAfter.rows ?? '?'} × {shapeAfter.cols ?? '?'}
            </div>
            <div style={{ fontSize: 11, color: '#2f54eb' }}>{isZh ? '行 × 列' : 'rows × cols'}</div>
          </div>
        </Col>
      </Row>

      {/* Metrics table */}
      <Table
        columns={columns}
        dataSource={dataSource}
        size="small"
        pagination={false}
        bordered={false}
        style={{ fontSize: 13 }}
        rowClassName={(_, idx) => idx % 2 === 0 ? '' : 'metrics-row-alt'}
      />
    </div>
  );
};

export default MetricsPanel;

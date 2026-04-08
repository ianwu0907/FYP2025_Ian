/**
 * 表格对比组件
 * 并排显示原始表格和标准化后的表格
 */

import React from 'react';
import { Row, Col, Card, Table, Alert, Tabs, Grid } from 'antd';
import { SwapOutlined } from '@ant-design/icons';

const { useBreakpoint } = Grid;
import { useLanguage } from '../../contexts/LanguageContext';
import './TableComparison.css';

const TableComparison = ({ originalData, normalizedData, singleMode = false }) => {
  const { t } = useLanguage();
  const [pageSize, setPageSize] = React.useState(10);
  const screens = useBreakpoint();
  const isMobile = !screens.md;

  // 将 preview 数据转换为 Ant Design Table 需要的格式
  const convertToTableData = (previewData) => {
    if (!previewData || !previewData.data || previewData.data.length === 0) {
      return { columns: [], dataSource: [] };
    }

    // 生成列配置
    const columns = previewData.columns.map((col, index) => ({
      title: col,
      dataIndex: col,
      key: col,
      ellipsis: true,
      width: 150,
      render: (text) => {
        // 处理 null/undefined
        if (text === null || text === undefined) {
          return <span style={{ color: '#999' }}>-</span>;
        }
        // 处理数字
        if (typeof text === 'number') {
          return text;
        }
        return String(text);
      },
    }));

    // 生成数据源（添加 key）
    const dataSource = previewData.data.map((row, index) => ({
      ...row,
      key: index,
    }));

    return { columns, dataSource };
  };

  const originalTable = convertToTableData(originalData);
  const normalizedTable = convertToTableData(normalizedData);

  // Single-mode: only show original table (used in error state)
  if (singleMode) {
    return (
      <div className="table-comparison">
        <Card
          title={t.tableComparison.original}
          bordered={false}
          headStyle={{ background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)', fontWeight: 'bold', color: '#667eea' }}
          bodyStyle={{ padding: 0 }}
          style={{ borderRadius: '12px', boxShadow: '0 4px 20px rgba(102, 126, 234, 0.15)', border: 'none' }}
        >
          <div className="table-wrapper">
            <Table
              columns={originalTable.columns}
              dataSource={originalTable.dataSource}
              pagination={{ pageSize, showSizeChanger: true, pageSizeOptions: ['10', '20', '50'], showTotal: (total) => t.tableComparison.totalRows.replace('{count}', total) }}
              scroll={{ x: 'max-content', y: 400 }}
              size="small"
              bordered
            />
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="table-comparison">
      <Alert
        message={t.tableComparison.title}
        description={
          t.tableComparison.dimensions
            .replace('{originalRows}', originalData?.shape?.[0] || 0)
            .replace('{originalCols}', originalData?.shape?.[1] || 0)
            .replace('{normalizedRows}', normalizedData?.shape?.[0] || 0)
            .replace('{normalizedCols}', normalizedData?.shape?.[1] || 0)
        }
        type="info"
        showIcon
        icon={<SwapOutlined />}
        style={{ marginBottom: 16 }}
      />

      {isMobile ? (
        /* 手机：Tabs 切换 */
        <Tabs
          defaultActiveKey="original"
          items={[
            {
              key: 'original',
              label: t.tableComparison.original,
              children: (
                <Table
                  columns={originalTable.columns}
                  dataSource={originalTable.dataSource}
                  pagination={{ pageSize, showSizeChanger: false, showTotal: (total) => t.tableComparison.totalRows.replace('{count}', total) }}
                  scroll={{ x: 'max-content', y: 350 }}
                  size="small"
                  bordered
                />
              ),
            },
            {
              key: 'normalized',
              label: t.tableComparison.normalized,
              children: normalizedData ? (
                <Table
                  columns={normalizedTable.columns}
                  dataSource={normalizedTable.dataSource}
                  pagination={{ pageSize, showSizeChanger: false, showTotal: (total) => t.tableComparison.totalRows.replace('{count}', total) }}
                  scroll={{ x: 'max-content', y: 350 }}
                  size="small"
                  bordered
                />
              ) : (
                <Alert message={t.tableComparison.waitingForResult} type="warning" />
              ),
            },
          ]}
        />
      ) : (
        /* 桌面：左右并排 */
        <Row gutter={16}>
          <Col span={12}>
            <Card
              title={t.tableComparison.original}
              bordered={false}
              headStyle={{ background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)', fontWeight: 'bold', color: '#667eea' }}
              bodyStyle={{ padding: 0 }}
              style={{ borderRadius: '12px', boxShadow: '0 4px 20px rgba(102, 126, 234, 0.15)', border: 'none' }}
            >
              <div className="table-wrapper">
                <Table
                  columns={originalTable.columns}
                  dataSource={originalTable.dataSource}
                  pagination={{ pageSize, showSizeChanger: true, pageSizeOptions: ['10', '20', '50', '100'], showTotal: (total) => t.tableComparison.totalRows.replace('{count}', total), onChange: (page, newPageSize) => { if (newPageSize !== pageSize) setPageSize(newPageSize); } }}
                  scroll={{ x: 'max-content', y: 400 }}
                  size="small"
                  bordered
                />
              </div>
            </Card>
          </Col>
          <Col span={12}>
            <Card
              title={t.tableComparison.normalized}
              bordered={false}
              headStyle={{ background: 'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)', fontWeight: 'bold', color: '#237804' }}
              bodyStyle={{ padding: 0 }}
              style={{ borderRadius: '12px', boxShadow: '0 4px 20px rgba(82, 196, 26, 0.15)', border: 'none' }}
            >
              <div className="table-wrapper">
                {normalizedData ? (
                  <Table
                    columns={normalizedTable.columns}
                    dataSource={normalizedTable.dataSource}
                    pagination={{ pageSize, showSizeChanger: true, pageSizeOptions: ['10', '20', '50', '100'], showTotal: (total) => t.tableComparison.totalRows.replace('{count}', total), onChange: (page, newPageSize) => { if (newPageSize !== pageSize) setPageSize(newPageSize); } }}
                    scroll={{ x: 'max-content', y: 400 }}
                    size="small"
                    bordered
                  />
                ) : (
                  <div style={{ padding: 20, textAlign: 'center' }}>
                    <Alert message={t.tableComparison.waitingForResult} type="warning" />
                  </div>
                )}
              </div>
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
};

export default TableComparison;

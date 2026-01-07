/**
 * 表格对比组件
 * 并排显示原始表格和标准化后的表格
 */

import React from 'react';
import { Row, Col, Card, Table, Alert } from 'antd';
import { SwapOutlined } from '@ant-design/icons';
import { useLanguage } from '../../contexts/LanguageContext';
import './TableComparison.css';

const TableComparison = ({ originalData, normalizedData }) => {
  const { t } = useLanguage();
  const [pageSize, setPageSize] = React.useState(10);

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

      <Row gutter={16}>
        {/* 原始表格 */}
        <Col span={12}>
          <Card
            title={t.tableComparison.original}
            bordered={false}
            headStyle={{ background: '#fafafa', fontWeight: 'bold' }}
            bodyStyle={{ padding: 0 }}
          >
            <div className="table-wrapper">
              <Table
                columns={originalTable.columns}
                dataSource={originalTable.dataSource}
                pagination={{
                  pageSize: pageSize,
                  showSizeChanger: true,
                  pageSizeOptions: ['10', '20', '50', '100'],
                  showTotal: (total) => t.tableComparison.totalRows.replace('{count}', total),
                  onChange: (page, newPageSize) => {
                    if (newPageSize !== pageSize) {
                      setPageSize(newPageSize);
                    }
                  },
                }}
                scroll={{ x: 'max-content', y: 400 }}
                size="small"
                bordered
              />
            </div>
          </Card>
        </Col>

        {/* 标准化后的表格 */}
        <Col span={12}>
          <Card
            title={t.tableComparison.normalized}
            bordered={false}
            headStyle={{ background: '#f6ffed', fontWeight: 'bold' }}
            bodyStyle={{ padding: 0 }}
          >
            <div className="table-wrapper">
              {normalizedData ? (
                <Table
                  columns={normalizedTable.columns}
                  dataSource={normalizedTable.dataSource}
                  pagination={{
                    pageSize: pageSize,
                    showSizeChanger: true,
                    pageSizeOptions: ['10', '20', '50', '100'],
                    showTotal: (total) => t.tableComparison.totalRows.replace('{count}', total),
                    onChange: (page, newPageSize) => {
                      if (newPageSize !== pageSize) {
                        setPageSize(newPageSize);
                      }
                    },
                  }}
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
    </div>
  );
};

export default TableComparison;

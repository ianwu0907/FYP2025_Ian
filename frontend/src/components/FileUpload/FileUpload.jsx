/**
 * 文件上传组件
 * 使用 Ant Design 的 Dragger 实现拖拽上传
 */

import React from 'react';
import { Upload } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

const { Dragger } = Upload;

const FileUpload = ({ onUpload, disabled, loading }) => {
  const props = {
    name: 'file',
    multiple: false,
    accept: '.xlsx,.xls,.csv',
    beforeUpload: (file) => {
      // 验证文件类型
      const isValidType = file.name.endsWith('.xlsx') ||
                          file.name.endsWith('.xls') ||
                          file.name.endsWith('.csv');

      if (!isValidType) {
        console.error('只支持 .xlsx, .xls 和 .csv 文件！');
        return false;
      }

      // 验证文件大小 (100MB)
      const isValidSize = file.size / 1024 / 1024 < 100;
      if (!isValidSize) {
        console.error('文件大小不能超过 100MB！');
        return false;
      }

      // 调用父组件的上传处理函数
      onUpload(file);

      // 阻止 antd 的默认上传行为
      return false;
    },
    disabled: disabled || loading,
    showUploadList: false, // 不显示上传列表
  };

  return (
    <Dragger {...props}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined style={{ fontSize: 48, color: '#1890ff' }} />
      </p>
      <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
      <p className="ant-upload-hint">
        支持 Excel (.xlsx, .xls) 和 CSV (.csv) 文件，最大 100MB
      </p>
    </Dragger>
  );
};

export default FileUpload;

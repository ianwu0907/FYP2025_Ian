/**
 * 文件上传组件
 * 使用 Ant Design 的 Dragger 实现拖拽上传
 */

import React from 'react';
import { Upload, message } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { useLanguage } from '../../contexts/LanguageContext';

const { Dragger } = Upload;

const FileUpload = ({ onUpload, disabled, loading }) => {
  const { t, currentLanguage } = useLanguage();
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
        const errorMsg = currentLanguage === 'zh'
          ? '只支持 .xlsx, .xls 和 .csv 文件！'
          : 'Only .xlsx, .xls and .csv files are supported!';
        message.error(errorMsg);
        return false;
      }

      // 验证文件大小 (100MB)
      const isValidSize = file.size / 1024 / 1024 < 100;
      if (!isValidSize) {
        const errorMsg = currentLanguage === 'zh'
          ? '文件大小不能超过 100MB！'
          : 'File size cannot exceed 100MB!';
        message.error(errorMsg);
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
      <p className="ant-upload-text">{t.upload.dragText}</p>
      <p className="ant-upload-hint">
        {t.upload.hint}
      </p>
    </Dragger>
  );
};

export default FileUpload;

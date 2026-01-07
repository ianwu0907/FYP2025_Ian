/**
 * 中文语言包
 */
export const zh = {
  // 通用
  common: {
    confirm: '确认',
    cancel: '取消',
    submit: '提交',
    reset: '重置',
    download: '下载',
    loading: '加载中...',
    success: '成功',
    error: '错误',
    warning: '警告',
  },

  // 页面标题
  header: {
    title: 'Spreadsheet Normalizer',
    subtitle: '表格标准化工具',
    restart: '重新开始',
  },

  // 步骤标题
  steps: {
    step1: '步骤 1: 上传文件',
    step2: '步骤 2: 开始标准化',
    step3: '步骤 3: 处理进度',
    step4: '步骤 4: 处理结果',
  },

  // 文件上传
  upload: {
    dragText: '点击或拖拽文件到此处上传',
    hint: '支持 Excel (.xlsx, .xls) 和 CSV 文件',
    uploading: '正在上传...',
    uploadSuccess: '文件上传成功！',
    uploadFailed: '上传失败',
    fileInfo: '文件信息',
    filename: '文件名',
    fileType: '文件类型',
    fileSize: '文件大小',
    dimensions: '数据维度',
    rows: '行',
    columns: '列',
    sessionId: 'Session ID',
  },

  // 标准化处理
  normalization: {
    start: '开始标准化处理',
    processing: '处理中...',
    completed: '处理完成',
    startSuccess: '标准化处理已开始！',
    startFailed: '启动失败',
  },

  // 进度显示
  progress: {
    title: '处理进度',
    completed: '已完成',
    processingLogs: '处理日志',
  },

  // 结果显示
  result: {
    title: '处理结果',
    outputPath: '输出路径',
    tableCount: '表格数量',
    detectionMethod: '检测方法',
    processingTime: '处理时间',
    seconds: '秒',
    downloadResult: '下载标准化结果',
    downloading: '正在准备下载...',
    downloadSuccess: '下载成功！',
    downloadFailed: '下载失败',
    tableComparison: '表格对比',
    comparisonWarning: '无法显示对比视图',
    normalizedPreviewExists: 'normalized_preview 存在',
    originalPreviewExists: 'uploadedFileInfo.preview 存在',
  },

  // 状态标签
  status: {
    idle: '等待上传',
    uploading: '正在上传...',
    uploaded: '上传完成',
    processing: '处理中...',
    completed: '处理完成',
    error: '错误',
  },

  // 错误消息
  error: {
    title: '错误信息',
    retry: '重新尝试',
    sessionNotFound: '会话未找到',
    taskNotFound: '任务未找到',
    networkError: '网络错误',
  },

  // 消息提示
  message: {
    resetSuccess: '已重置，可以上传新文件',
    pollingFailed: '轮询失败（已重试5次）',
    taskNotExists: '任务不存在（可能后端已重启）',
    connectionFailed: '连接失败',
  },

  // 表格对比
  tableComparison: {
    title: '表格对比',
    original: '原始表格',
    normalized: '标准化后的表格',
    totalRows: '共 {count} 行',
    dimensions: '原始表格: {originalRows} 行 × {originalCols} 列 | 标准化后: {normalizedRows} 行 × {normalizedCols} 列',
    waitingForResult: '处理完成后将显示标准化结果',
  },
};

export default zh;

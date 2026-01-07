/**
 * English Language Pack
 */
export const en = {
  // Common
  common: {
    confirm: 'Confirm',
    cancel: 'Cancel',
    submit: 'Submit',
    reset: 'Reset',
    download: 'Download',
    loading: 'Loading...',
    success: 'Success',
    error: 'Error',
    warning: 'Warning',
  },

  // Header
  header: {
    title: 'Spreadsheet Normalizer',
    subtitle: 'Spreadsheet Normalization Tool',
    restart: 'Restart',
  },

  // Steps
  steps: {
    step1: 'Step 1: Upload File',
    step2: 'Step 2: Start Normalization',
    step3: 'Step 3: Processing Progress',
    step4: 'Step 4: Results',
  },

  // File Upload
  upload: {
    dragText: 'Click or drag file to this area to upload',
    hint: 'Support Excel (.xlsx, .xls) and CSV files',
    uploading: 'Uploading...',
    uploadSuccess: 'File uploaded successfully!',
    uploadFailed: 'Upload failed',
    fileInfo: 'File Information',
    filename: 'Filename',
    fileType: 'File Type',
    fileSize: 'File Size',
    dimensions: 'Dimensions',
    rows: 'rows',
    columns: 'columns',
    sessionId: 'Session ID',
  },

  // Normalization
  normalization: {
    start: 'Start Normalization',
    processing: 'Processing...',
    completed: 'Completed',
    startSuccess: 'Normalization started!',
    startFailed: 'Failed to start',
  },

  // Progress
  progress: {
    title: 'Processing Progress',
    completed: 'Completed',
    processingLogs: 'Processing Logs',
  },

  // Results
  result: {
    title: 'Results',
    outputPath: 'Output Path',
    tableCount: 'Table Count',
    detectionMethod: 'Detection Method',
    processingTime: 'Processing Time',
    seconds: 'seconds',
    downloadResult: 'Download Normalized Result',
    downloading: 'Preparing download...',
    downloadSuccess: 'Download successful!',
    downloadFailed: 'Download failed',
    tableComparison: 'Table Comparison',
    comparisonWarning: 'Cannot display comparison view',
    normalizedPreviewExists: 'normalized_preview exists',
    originalPreviewExists: 'uploadedFileInfo.preview exists',
  },

  // Status Labels
  status: {
    idle: 'Waiting for upload',
    uploading: 'Uploading...',
    uploaded: 'Uploaded',
    processing: 'Processing...',
    completed: 'Completed',
    error: 'Error',
  },

  // Error Messages
  error: {
    title: 'Error Information',
    retry: 'Retry',
    sessionNotFound: 'Session not found',
    taskNotFound: 'Task not found',
    networkError: 'Network error',
  },

  // Messages
  message: {
    resetSuccess: 'Reset successful, you can upload a new file',
    pollingFailed: 'Polling failed (retried 5 times)',
    taskNotExists: 'Task does not exist (backend may have restarted)',
    connectionFailed: 'Connection failed',
  },

  // Table Comparison
  tableComparison: {
    title: 'Table Comparison',
    original: 'Original Table',
    normalized: 'Normalized Table',
    totalRows: 'Total {count} rows',
    dimensions: 'Original: {originalRows} rows × {originalCols} columns | Normalized: {normalizedRows} rows × {normalizedCols} columns',
    waitingForResult: 'Normalized result will be displayed after processing',
  },
};

export default en;

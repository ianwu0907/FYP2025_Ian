/**
 * App 主组件
 */

import React from 'react';
import { LanguageProvider } from './contexts/LanguageContext';
import NormalizerPage from './pages/NormalizerPage';
import './App.css';

function App() {
  return (
    <LanguageProvider>
      <NormalizerPage />
    </LanguageProvider>
  );
}

export default App;

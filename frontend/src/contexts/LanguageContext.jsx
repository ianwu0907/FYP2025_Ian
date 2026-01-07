/**
 * 语言上下文
 * 管理应用的多语言状态
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import zh from '../locales/zh';
import en from '../locales/en';

const LanguageContext = createContext();

const languages = {
  zh,
  en,
};

export const LanguageProvider = ({ children }) => {
  // 从 localStorage 读取保存的语言，默认为中文
  const [currentLanguage, setCurrentLanguage] = useState(() => {
    const saved = localStorage.getItem('language');
    return saved || 'zh';
  });

  // 当语言改变时，保存到 localStorage
  useEffect(() => {
    localStorage.setItem('language', currentLanguage);
  }, [currentLanguage]);

  // 切换语言
  const toggleLanguage = () => {
    setCurrentLanguage((prev) => (prev === 'zh' ? 'en' : 'zh'));
  };

  // 设置语言
  const setLanguage = (lang) => {
    if (languages[lang]) {
      setCurrentLanguage(lang);
    }
  };

  // 获取当前语言的翻译对象
  const t = languages[currentLanguage];

  const value = {
    currentLanguage,
    toggleLanguage,
    setLanguage,
    t,
    isZh: currentLanguage === 'zh',
    isEn: currentLanguage === 'en',
  };

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
};

// 自定义 Hook
export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

export default LanguageContext;

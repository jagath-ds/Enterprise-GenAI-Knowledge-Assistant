import React from 'react';
import { Routes, Route } from 'react-router-dom';

// Existing Employee Pages
import LoginPage from './pages/LoginPage';
import Dashboard from './pages/Dashboard';

// New Admin Pages
import AdminLogin from './pages/AdminLogin';
import AdminPortal from './pages/AdminPortal';

export default function App() {
  return (
    <Routes>
      {/* Normal Employee Routes */}
      <Route path="/" element={<LoginPage />} />
      <Route path="/dashboard" element={<Dashboard />} />
      
      {/* Hidden Admin Routes */}
      <Route path="/secret-admin" element={<AdminLogin />} />
      <Route path="/admin-dashboard" element={<AdminPortal />} />
    </Routes>
  );
}
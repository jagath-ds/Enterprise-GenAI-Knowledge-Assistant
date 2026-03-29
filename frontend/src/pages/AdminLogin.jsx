import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldAlert, KeyRound } from 'lucide-react';
import { adminLogin } from '../services/api';

export default function AdminLogin() {
const [email, setEmail] = useState('');
const [password, setPassword] = useState('');
const [error, setError] = useState('');
const navigate = useNavigate();
const [loading, setLoading] = useState(false);
const handleAdminLogin = async (e) => {
e.preventDefault();
setError('');
setLoading(true);

try {
  const data = await adminLogin(email, password);

  if (!data.access_token) {
    setError(data.detail || "Invalid credentials");
    return;
  }

  // ✅ Store JWT
  localStorage.setItem("token", data.access_token);
  localStorage.setItem("role", data.role);
  localStorage.setItem("user_id", data.user_id);
  localStorage.setItem("user_name", data.name);

  // ✅ Redirect to admin dashboard
  navigate('/admin-dashboard');

} catch (err) {
  console.error(err);
  setError("Server error");
} finally {
  setLoading(false);
}

};

return (
  <div className="min-h-screen flex items-center justify-center bg-zinc-950 text-gray-200 font-sans">
    {" "}
    <div className="bg-zinc-900 p-10 rounded-2xl shadow-2xl w-full max-w-md border border-zinc-800 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-1 bg-red-600"></div>

      <div className="flex flex-col items-center mb-8">
        <div className="p-3 bg-red-500/10 rounded-full mb-3 border border-red-500/20">
          <ShieldAlert size={32} className="text-red-500" />
        </div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          Admin Gateway
        </h2>
        <p className="text-xs text-zinc-500 mt-2 uppercase tracking-widest">
          Restricted Area
        </p>
      </div>

      <form onSubmit={handleAdminLogin} className="space-y-6">
        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-red-400 text-xs font-mono text-center">
            {error}
          </div>
        )}

        <div className="space-y-2">
          <label className="block text-xs font-bold text-zinc-400 uppercase tracking-wider">
            Admin Email
          </label>
          <input
            type="email"
            className="w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg focus:ring-2 focus:ring-red-500 outline-none transition text-sm text-white"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>

        <div className="space-y-2">
          <label className="block text-xs font-bold text-zinc-400 uppercase tracking-wider">
            Master Key
          </label>
          <input
            type="password"
            className="w-full p-3 bg-zinc-950 border border-zinc-800 rounded-lg focus:ring-2 focus:ring-red-500 outline-none transition text-sm text-white"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-red-600 hover:bg-red-700 text-white font-bold p-3 rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-50"
        >
          <KeyRound size={18} />
          {loading ? "Authenticating..." : "Authenticate"}
        </button>
      </form>
    </div>
  </div>
);
}

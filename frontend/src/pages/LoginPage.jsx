import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, ShieldCheck, HelpCircle, UserPlus, LogIn, CheckCircle2 } from 'lucide-react';
import { employeeLogin } from '../services/api';
import { registerUser } from '../services/api';
export default function LoginPage() {
const [email, setEmail] = useState('');
const [password, setPassword] = useState('');
const [name, setName] = useState('');
const [isRegistering, setIsRegistering] = useState(false);
const [error, setError] = useState('');
const [successMsg, setSuccessMsg] = useState('');
const [showHelp, setShowHelp] = useState(false);

const navigate = useNavigate();

const handleAuth = async (e) => {
e.preventDefault();
setError('');
setSuccessMsg('');
setShowHelp(false);

try {
 if (isRegistering) {
   // split name
   const nameParts = name.trim().split(" ");
   const firstname = nameParts[0];
   const lastname = nameParts.slice(1).join(" ") || "User";

   const data = await registerUser(firstname, lastname, email, password);

   if (data.detail) {
     setError(data.detail);
     return;
   }

   setSuccessMsg("Registration successful. Await admin approval.");
   return;
 }

  // 🔐 LOGIN (backend)
  const data = await employeeLogin(email, password);

  if (data.detail) {
    setError(data.detail);
    return;
  }

  // ✅ Store JWT
  localStorage.setItem("token", data.access_token);
  localStorage.setItem("role", data.role);
  localStorage.setItem("user_id", data.user_id);
  localStorage.setItem("user_name", data.name);
  
  // ✅ Redirect
  navigate('/dashboard');

} catch (err) {
  console.error(err);
  setError("Server error");
}

};

return ( <div className="min-h-screen flex items-center justify-center bg-gray-50 text-gray-900 font-sans"> <div className="bg-white p-10 rounded-2xl shadow-xl w-full max-w-md border border-gray-100 relative">

    {/* Header */}
    <div className="flex flex-col items-center mb-8">
      <div className="p-3 bg-blue-50 rounded-full mb-3">
        <ShieldCheck size={32} className="text-blue-600" />
      </div>
      <h2 className="text-3xl font-bold text-gray-900 tracking-tight">AI Assistant</h2>
      <p className="text-sm text-gray-500 mt-2">
        {isRegistering ? 'Employee Registration' : 'Enterprise Access Portal'}
      </p>
    </div>
    
    <form onSubmit={handleAuth} className="space-y-5">
      
      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 p-3 bg-red-50 border border-red-100 rounded-lg text-red-600 text-sm">
          <AlertCircle size={18} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Success */}
      {successMsg && (
        <div className="flex items-start gap-3 p-3 bg-green-50 border border-green-100 rounded-lg text-green-700 text-sm">
          <CheckCircle2 size={18} className="mt-0.5 flex-shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}

      {isRegistering && (
        <div className="space-y-2">
          <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider">
            Full Name
          </label>
          <input 
            type="text" 
            className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition text-sm" 
            placeholder="John Doe"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>
      )}

      <div className="space-y-2">
        <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider">
          Company Email
        </label>
        <input 
          type="email" 
          className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition text-sm" 
          placeholder="name@company.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>
      
      <div className="space-y-2">
        <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider">
          Password
        </label>
        <input 
          type="password" 
          className="w-full p-3 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none transition text-sm font-sans"
          placeholder="••••••••••"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>
      
      <button 
        type="submit" 
        className="w-full bg-blue-600 text-white font-bold p-3 rounded-lg hover:bg-blue-700 shadow-md transition-all active:scale-[0.98] mt-2 flex items-center justify-center gap-2"
      >
        {isRegistering 
          ? <><UserPlus size={18}/> Request Access</> 
          : <><LogIn size={18}/> Secure Sign In</>}
      </button>
    </form>
    
    {/* Footer */}
    <div className="mt-8 text-center space-y-4">
      <button 
        onClick={() => {
          setIsRegistering(!isRegistering);
          setError('');
          setSuccessMsg('');
        }}
        className="text-xs text-blue-600 hover:text-blue-800 font-medium transition"
      >
        {isRegistering 
          ? 'Already have an account? Sign In' 
          : 'New employee? Request access here'}
      </button>

      {!isRegistering && (
        <button 
          onClick={() => setShowHelp(!showHelp)}
          className="text-xs text-gray-400 hover:text-gray-600 font-medium flex items-center justify-center gap-1 mx-auto"
        >
          <HelpCircle size={14} /> Trouble logging in?
        </button>
      )}

      {showHelp && (
        <div className="text-xs text-gray-500 bg-gray-50 p-3 rounded-lg border border-gray-200">
          Contact <strong>IT Support</strong>.
        </div>
      )}

      <div className="flex items-center justify-center gap-2 opacity-50">
        <div className="h-px bg-gray-300 w-12"></div>
        <span className="text-[10px] text-gray-400 uppercase tracking-widest">
          Authorized Use Only
        </span>
        <div className="h-px bg-gray-300 w-12"></div>
      </div>
    </div>
  </div>
</div>
);
}

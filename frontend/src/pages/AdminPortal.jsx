import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  uploadDocuments,
  getDocuments,
  deleteDocument,
  reindexDocument,
  getEmployees,
  approveUser,
  updateUserStatus,
  deleteUser,
  registerUser,
  getLogs,
} from "../services/api";
import { StrokxLogoFull } from "../components/StrokxLogo";

// ─── Toast Component ─────────────────────────────────────────────────────────
function Toast({ toasts, removeToast }) {
  return (
    <div className="fixed top-5 right-5 z-[9999] flex flex-col gap-2.5">
      {toasts.map((t) => (
        <div
          key={t.id}
          onClick={() => removeToast(t.id)}
          className={`px-4 py-3 rounded-xl text-white text-sm font-medium shadow-lg cursor-pointer min-w-[220px] animate-[slideIn_0.3s_ease] ${
            t.type === "success"
              ? "bg-green-500"
              : t.type === "error"
                ? "bg-red-500"
                : "bg-blue-500"
          }`}
        >
          {t.type === "success" ? "✓ " : t.type === "error" ? "✕ " : "ℹ "}
          {t.message}
        </div>
      ))}
    </div>
  );
}

// ─── Confirm Modal ────────────────────────────────────────────────────────────
function ConfirmModal({ open, message, onConfirm, onCancel, danger = false }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/40 z-[8000] flex items-center justify-center">
      <div className="bg-white rounded-2xl p-8 max-w-sm w-[90%] shadow-2xl">
        <p className="text-[15px] text-slate-700 mb-7 leading-relaxed">
          {message}
        </p>
        <div className="flex gap-2.5 justify-end">
          <button
            onClick={onCancel}
            className="px-5 py-2 rounded-lg border border-slate-200 bg-white cursor-pointer text-sm text-slate-500 hover:bg-slate-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-5 py-2 rounded-lg border-none cursor-pointer text-sm font-semibold text-white transition-colors ${
              danger
                ? "bg-red-500 hover:bg-red-600"
                : "bg-slate-900 hover:bg-slate-800"
            }`}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Status Badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  const map = {
    active: { classes: "bg-green-100 text-green-700", label: "Active" },
    suspended: { classes: "bg-amber-100 text-amber-700", label: "Suspended" },
    pending: { classes: "bg-blue-100 text-blue-700", label: "Pending" },
    success: { classes: "bg-green-100 text-green-700", label: "Success" },
    failed: { classes: "bg-red-100 text-red-700", label: "Failed" },
    uploaded: { classes: "bg-slate-100 text-slate-700", label: "Uploaded" },
    indexing: {
      classes: "bg-yellow-100 text-yellow-700",
      label: "Indexing...",
    },
    indexed: { classes: "bg-sky-100 text-sky-700", label: "Indexed" },
    Indexed: { classes: "bg-sky-100 text-sky-700", label: "Indexed" },
    "Reindexing...": {
      classes: "bg-yellow-100 text-yellow-700",
      label: "Reindexing...",
    },
  };
  const s = map[status] || {
    classes: "bg-slate-100 text-slate-700",
    label: status,
  };
  return (
    <span
      className={`px-2.5 py-1 rounded-full text-xs font-semibold tracking-wide ${s.classes}`}
    >
      {s.label}
    </span>
  );
}

// ─── Log Action Badge ─────────────────────────────────────────────────────────
function ActionBadge({ action }) {
  const map = {
    LOGIN: "bg-blue-100 text-blue-700",
    LOGOUT: "bg-slate-100 text-slate-700",
    UPLOAD: "bg-green-100 text-green-700",
    DELETE: "bg-red-100 text-red-700",
    QUERY: "bg-purple-100 text-purple-700",
  };
  const classes = map[action] || "bg-slate-100 text-slate-700";
  return (
    <span
      className={`px-2 py-0.5 rounded-md text-[11px] font-bold tracking-wide whitespace-nowrap ${classes}`}
    >
      {action}
    </span>
  );
}

export default function AdminPortal() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState("dashboard");
  const [showAddModal, setShowAddModal] = useState(false);
  const [newEmp, setNewEmp] = useState({ name: "", email: "", password: "" });
  const [files, setFiles] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [logs, setLogs] = useState([]);
  const [users, setUsers] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [confirm, setConfirm] = useState({
    open: false,
    message: "",
    onConfirm: null,
    danger: false,
  });
  const [logFilter, setLogFilter] = useState("ALL");
  const [userSearch, setUserSearch] = useState("");
  const [addingUser, setAddingUser] = useState(false);
  const toastId = useRef(0);
  const [duplicateFile, setDuplicateFile] = useState(null);
  const addToast = (message, type = "success") => {
    const id = ++toastId.current;
    setToasts((p) => [...p, { id, message, type }]);
    setTimeout(() => removeToast(id), 3500);
  };
  const removeToast = (id) => setToasts((p) => p.filter((t) => t.id !== id));

  const askConfirm = (message, onConfirm, danger = false) =>
    setConfirm({ open: true, message, onConfirm, danger });

  useEffect(() => {
    const token = localStorage.getItem("token");
    const role = localStorage.getItem("role");
    if (!token || role !== "admin") navigate("/secret-admin");
  }, [navigate]);

  useEffect(() => {
    fetchDocs();
    fetchUsers();
    fetchLogs();

    const interval = setInterval(() => {
      fetchDocs();
      fetchLogs();
    }, 10000); // every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const isDocumentBusy = (status) =>
    status === "indexing" || status === "Reindexing...";

  const fetchDocs = async () => {
    const data = await getDocuments();
    setDocuments(
      data.map((doc) => ({
        id: doc.id,
        name: doc.filename,
        status: doc.status || "uploaded",
        uploadedAt: doc.created_at || null,
        size: doc.size || null,
      })),
    );
  };

  const fetchUsers = async () => {
    const data = await getEmployees();
    setUsers(data);
  };

  const fetchLogs = async () => {
    const data = await getLogs();
    setLogs(Array.isArray(data) ? data : []);
  };

  const handleUpload = async () => {
  if (files.length === 0) return;
  setUploading(true);
  try {
    await uploadDocuments(files);
    await fetchDocs();
    await fetchLogs();
    setFiles([]);
    addToast(`${files.length} document(s) uploaded successfully`);
  } catch (err) {
    const msg = err.message || "";
    if (msg.includes("already exists")) {
      // Extract filename from error message
      const match = msg.match(/'(.+?)'/);
      const conflictName = match ? match[1] : files[0].name;
      setDuplicateFile({ file: files[0], name: conflictName });
    } else {
      addToast("Upload failed. Please try again.", "error");
    }
  }
  setUploading(false);
};

const handleOverwrite = async () => {
  if (!duplicateFile) return;
  setUploading(true);
  try {
    // Delete the existing doc first, then re-upload
    const existing = documents.find((d) => d.name === duplicateFile.name);
    if (existing) await deleteDocument(existing.id);
    await uploadDocuments([duplicateFile.file]);
    await fetchDocs();
    await fetchLogs();
    setFiles([]);
    addToast(`"${duplicateFile.name}" overwritten successfully`);
  } catch {
    addToast("Overwrite failed. Please try again.", "error");
  }
  setDuplicateFile(null);
  setUploading(false);
};

  const handleDelete = (id, name) => {
    const document = documents.find((doc) => doc.id === id);
    if (document && isDocumentBusy(document.status)) {
      addToast("This document is still indexing. Please wait.", "info");
      return;
    }

    askConfirm(
      `Delete "${name}"? This cannot be undone.`,
      async () => {
        setConfirm((p) => ({ ...p, open: false }));
        await deleteDocument(id);
        setDocuments((prev) => prev.filter((doc) => doc.id !== id));
        await fetchLogs();
        addToast("Document deleted");
      },
      true,
    );
  };

  const handleReindex = (id) => {
    const document = documents.find((doc) => doc.id === id);
    if (document && isDocumentBusy(document.status)) {
      addToast("This document is already indexing. Please wait.", "info");
      return;
    }

    setDocuments((prev) =>
      prev.map((doc) =>
        doc.id === id ? { ...doc, status: "Reindexing..." } : doc,
      ),
    );
    setTimeout(async () => {
      await reindexDocument(id);
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.id === id ? { ...doc, status: "Indexed" } : doc,
        ),
      );
      await fetchLogs();
      addToast("Document reindexed");
    }, 1200);
  };

  const handleApprove = async (id) => {
    await approveUser(id);
    await fetchUsers();
    await fetchLogs();
    addToast("User approved");
  };

  const handleStatusChange = (id, status, name) => {
    askConfirm(
      `${status === "suspended" ? "Suspend" : "Activate"} user "${name}"?`,
      async () => {
        setConfirm((p) => ({ ...p, open: false }));
        await updateUserStatus(id, status);
        await fetchUsers();
        await fetchLogs();
        addToast(`User ${status === "suspended" ? "suspended" : "activated"}`);
      },
      status === "suspended",
    );
  };

  const handleDeleteUser = (id, name) => {
    askConfirm(
      `Permanently delete user "${name}"?`,
      async () => {
        setConfirm((p) => ({ ...p, open: false }));
        await deleteUser(id);
        await fetchUsers();
        await fetchLogs();
        addToast("User deleted", "info");
      },
      true,
    );
  };

  const handleAddEmployee = async (e) => {
    e.preventDefault();
    setAddingUser(true);
    try {
      await registerUser({
        firstname: newEmp.name.split(" ")[0],
        lastname: newEmp.name.split(" ")[1] || "",
        email: newEmp.email,
        password: newEmp.password,
      });
      await fetchUsers();
      await fetchLogs();
      setNewEmp({ name: "", email: "", password: "" });
      setShowAddModal(false);
      addToast("User created successfully");
    } catch {
      addToast("Failed to create user", "error");
    }
    setAddingUser(false);
  };

  const handleLogout = async () => {
    const token = localStorage.getItem("token");
    await fetch("http://localhost:8000/auth/logout", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        user_id: localStorage.getItem("user_id"),
        role: localStorage.getItem("role"),
      }),
    });
    localStorage.clear();
    navigate("/secret-admin");
  };

  const filteredLogs =
    logFilter === "ALL" ? logs : logs.filter((l) => l.action === logFilter);
  const filteredUsers = users.filter(
    (u) =>
      u.name?.toLowerCase().includes(userSearch.toLowerCase()) ||
      u.email?.toLowerCase().includes(userSearch.toLowerCase()),
  );

  const statCards = [
    {
      label: "Documents",
      value: documents.length,
      icon: "📄",
      bg: "bg-indigo-50",
      text: "text-indigo-600",
    },
    {
      label: "Users",
      value: users.length,
      icon: "👥",
      bg: "bg-sky-50",
      text: "text-sky-600",
    },
    {
      label: "Total Logs",
      value: logs.length,
      icon: "📋",
      bg: "bg-emerald-50",
      text: "text-emerald-600",
    },
    {
      label: "Active Users",
      value: users.filter((u) => u.status === "active").length,
      icon: "✅",
      bg: "bg-amber-50",
      text: "text-amber-600",
    },
  ];

  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: "⊞" },
    { id: "documents", label: "Documents", icon: "📁" },
    { id: "employees", label: "Users", icon: "👤" },
    { id: "logs", label: "Logs", icon: "📋" },
  ];

  const uniqueActions = [
    "ALL",
    ...Array.from(new Set(logs.map((l) => l.action).filter(Boolean))),
  ];

  return (
    <>
      {/* Keeping just the custom keyframes to avoid altering tailwind.config.js */}
      <style>{`
        @keyframes slideIn { from { opacity:0; transform: translateX(20px); } to { opacity:1; transform: translateX(0); } }
        @keyframes fadeUp { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform: translateY(0); } }
        .fade-up { animation: fadeUp 0.35s ease both; }
      `}</style>

      <Toast toasts={toasts} removeToast={removeToast} />
      <ConfirmModal
        {...confirm}
        onCancel={() => setConfirm((p) => ({ ...p, open: false }))}
      />

      <div className="flex h-screen bg-slate-50 overflow-hidden font-sans">
        {/* ── Sidebar ── */}
        <aside className="w-60 bg-gradient-to-b from-slate-900 to-slate-800 flex flex-col shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.15)]">
          {/* Logo */}
          <div className="pt-7 pb-5 px-6 border-b border-white/5">
            <StrokxLogoFull isDark={true} />
            <div className="text-xs text-slate-400 mt-3 font-medium">
              {localStorage.getItem("user_name") || "Admin User"}
            </div>
            <div className="mt-2.5 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-green-500/15 border border-green-500/25">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block"></span>
              <span className="text-[11px] text-green-300 font-semibold">
                Online
              </span>
            </div>
          </div>

          {/* Nav */}
          <nav className="p-4 flex-1">
            <div className="text-[10px] text-slate-500 font-bold tracking-widest px-3 mb-2">
              NAVIGATION
            </div>
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full px-3.5 py-2.5 rounded-lg border-none flex items-center gap-2.5 cursor-pointer text-left text-sm transition-all duration-200 mb-0.5 hover:bg-white/10 hover:translate-x-1 ${
                  activeTab === item.id
                    ? "bg-white/15 text-white font-semibold"
                    : "bg-transparent text-slate-400 font-normal"
                }`}
              >
                <span className="text-base">{item.icon}</span>
                {item.label}
                {item.id === "employees" &&
                  users.filter((u) => u.status === "pending").length > 0 && (
                    <span className="ml-auto bg-red-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full">
                      {users.filter((u) => u.status === "pending").length}
                    </span>
                  )}
              </button>
            ))}
          </nav>

          {/* Logout */}
          <div className="p-4 border-t border-white/5">
            <button
              onClick={handleLogout}
              className="w-full px-3.5 py-2.5 rounded-lg border border-red-500/25 bg-red-500/10 text-red-400 cursor-pointer text-sm font-medium flex items-center gap-2 transition-all hover:bg-red-500/20"
            >
              <span>⎋</span> Logout
            </button>
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="flex-1 overflow-auto p-8">
          {/* ── DASHBOARD ── */}
          {activeTab === "dashboard" && (
            <div className="fade-up">
              <div className="mb-7">
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                  Dashboard
                </h1>
                <p className="text-sm text-slate-500 mt-1">
                  System overview —{" "}
                  {new Date().toLocaleDateString("en-US", {
                    weekday: "long",
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </p>
              </div>

              {/* Stat cards */}
              <div className="grid grid-cols-4 gap-4 mb-7">
                {statCards.map((s) => (
                  <div
                    key={s.label}
                    className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100 flex items-center gap-4 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md"
                  >
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center text-2xl shrink-0 ${s.bg} ${s.text}`}
                    >
                      {s.icon}
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-slate-900 leading-none">
                        {s.value}
                      </div>
                      <div className="text-[13px] text-slate-500 mt-1 font-medium">
                        {s.label}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Recent logs preview */}
              <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-[15px] font-bold text-slate-900">
                    Recent Activity
                  </h2>
                  <button
                    onClick={() => setActiveTab("logs")}
                    className="text-xs text-indigo-500 bg-transparent border-none cursor-pointer font-semibold hover:text-indigo-600"
                  >
                    View all →
                  </button>
                </div>
                {logs.slice(0, 5).map((log) => (
                  <div
                    key={log.id}
                    className="flex items-center gap-3 py-2.5 border-b border-slate-100 last:border-0"
                  >
                    <ActionBadge action={log.action} />
                    <span className="text-[13px] text-slate-700 flex-1">
                      {log.description}
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">
                      {log.time?.slice(11, 19)}
                    </span>
                  </div>
                ))}
                {logs.length === 0 && (
                  <p className="text-[13px] text-slate-400">No activity yet.</p>
                )}
              </div>
            </div>
          )}

          {/* ── DOCUMENTS ── */}
          {activeTab === "documents" && (
            <div className="fade-up">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                    Documents
                  </h1>
                  <p className="text-sm text-slate-500 mt-1">
                    {documents.length} document
                    {documents.length !== 1 ? "s" : ""} indexed
                  </p>
                </div>
              </div>

              {/* Upload zone */}
              <div className="bg-white rounded-2xl p-5 mb-5 border-[1.5px] border-dashed border-slate-300 shadow-sm flex items-center gap-4 flex-wrap">
                <label className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-50 border-[1.5px] border-slate-200 cursor-pointer text-[13px] font-semibold text-slate-700 hover:bg-slate-100 transition-colors">
                  📎 Choose Files
                  <input
                    type="file"
                    multiple
                    onChange={(e) => setFiles(Array.from(e.target.files))}
                    className="hidden"
                  />
                </label>
                {files.length > 0 && (
                  <span className="text-[13px] text-indigo-500 font-medium">
                    {files.length} file{files.length !== 1 ? "s" : ""} selected
                  </span>
                )}
                <button
                  onClick={handleUpload}
                  disabled={uploading || files.length === 0}
                  className={`px-5 py-2 rounded-lg border-none text-[13px] font-semibold transition-all ${
                    files.length === 0
                      ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                      : "bg-indigo-500 text-white cursor-pointer hover:bg-indigo-600 hover:-translate-y-px hover:shadow-md"
                  }`}
                >
                  {uploading ? "⏳ Uploading..." : "⬆ Upload"}
                </button>
              </div>

              {/* Document list */}
              <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-slate-100">
                {documents.length === 0 ? (
                  <div className="py-16 px-8 text-center">
                    <div className="text-4xl mb-3">📭</div>
                    <p className="text-[15px] text-slate-500 font-medium">
                      No documents uploaded yet
                    </p>
                    <p className="text-[13px] text-slate-400 mt-1">
                      Upload a PDF or DOCX to get started
                    </p>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-[1fr_120px_180px] px-5 py-3 border-b border-slate-100 text-[11px] font-bold text-slate-400 tracking-wider">
                      <span>FILENAME</span>
                      <span>STATUS</span>
                      <span className="text-right">ACTIONS</span>
                    </div>
                    {documents.map((doc, i) => (
                      <div
                        key={doc.id}
                        className={`grid grid-cols-[1fr_120px_180px] px-5 py-3.5 items-center transition-colors hover:bg-slate-50 ${i < documents.length - 1 ? "border-b border-slate-50" : ""}`}
                      >
                        <div className="flex items-center gap-2.5">
                          <span className="text-xl">📄</span>
                          <span className="text-[13px] font-medium text-slate-800">
                            {doc.name}
                          </span>
                        </div>
                        <StatusBadge status={doc.status} />
                        <div className="flex gap-2 justify-end">
                          <button
                            onClick={() => handleReindex(doc.id)}
                            disabled={isDocumentBusy(doc.status)}
                            className={`px-3 py-1.5 rounded-md border-[1.5px] border-indigo-100 bg-indigo-50 text-indigo-500 text-xs font-semibold transition-colors ${
                              isDocumentBusy(doc.status)
                                ? "cursor-not-allowed opacity-60"
                                : "cursor-pointer hover:bg-indigo-100"
                            }`}
                          >
                            ↺ Reindex
                          </button>
                          <button
                            onClick={() => handleDelete(doc.id, doc.name)}
                            disabled={isDocumentBusy(doc.status)}
                            className={`px-3 py-1.5 rounded-md border-[1.5px] border-red-100 bg-red-50 text-red-500 text-xs font-semibold transition-colors ${
                              isDocumentBusy(doc.status)
                                ? "cursor-not-allowed opacity-60"
                                : "cursor-pointer hover:bg-red-100"
                            }`}
                          >
                            🗑 Delete
                          </button>
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>
            </div>
          )}

          {/* ── USERS ── */}
          {activeTab === "employees" && (
            <div className="fade-up">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                    Users
                  </h1>
                  <p className="text-sm text-slate-500 mt-1">
                    {users.length} total ·{" "}
                    {users.filter((u) => u.status === "active").length} active ·{" "}
                    {users.filter((u) => u.status === "pending").length} pending
                  </p>
                </div>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="px-5 py-2.5 rounded-xl border-none bg-slate-900 text-white text-sm font-semibold flex items-center gap-2 cursor-pointer transition-all hover:bg-slate-800 hover:-translate-y-px shadow-sm"
                >
                  + Add User
                </button>
              </div>

              {/* Search */}
              <div className="mb-4">
                <input
                  placeholder="🔍  Search by name or email..."
                  value={userSearch}
                  onChange={(e) => setUserSearch(e.target.value)}
                  className="w-full max-w-[380px] px-4 py-2.5 rounded-xl border-[1.5px] border-slate-200 bg-white text-[13px] text-slate-700 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-shadow"
                />
              </div>

              {filteredUsers.length === 0 ? (
                <div className="bg-white rounded-2xl py-16 px-8 text-center shadow-sm">
                  <div className="text-4xl mb-3">👤</div>
                  <p className="text-[15px] text-slate-500 font-medium">
                    No users found
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-2.5">
                  {filteredUsers.map((user) => (
                    <div
                      key={user.id}
                      className="bg-white rounded-2xl px-5 py-4 shadow-sm border border-slate-100 flex justify-between items-center transition-all hover:-translate-y-0.5 hover:shadow-md"
                    >
                      <div className="flex items-center gap-3.5">
                        <div className="w-10 h-10 rounded-full bg-indigo-50 flex items-center justify-center text-base font-bold text-indigo-600">
                          {(user.name || "U")[0].toUpperCase()}
                        </div>
                        <div>
                          <p className="font-semibold text-[14px] text-slate-800">
                            {user.name}
                          </p>
                          <p className="text-xs text-slate-400 mt-0.5">
                            {user.email}
                          </p>
                        </div>
                        <StatusBadge status={user.status} />
                      </div>

                      <div className="flex gap-2">
                        {user.status === "pending" && (
                          <button
                            onClick={() => handleApprove(user.id)}
                            className="px-3.5 py-1.5 rounded-lg border-none bg-green-500 text-white text-xs font-semibold hover:bg-green-600 transition-colors cursor-pointer"
                          >
                            ✓ Approve
                          </button>
                        )}
                        {user.status === "active" && (
                          <button
                            onClick={() =>
                              handleStatusChange(
                                user.id,
                                "suspended",
                                user.name,
                              )
                            }
                            className="px-3.5 py-1.5 rounded-lg border-[1.5px] border-amber-200 bg-amber-50 text-amber-700 text-xs font-semibold hover:bg-amber-100 transition-colors cursor-pointer"
                          >
                            ⏸ Suspend
                          </button>
                        )}
                        {user.status === "suspended" && (
                          <button
                            onClick={() =>
                              handleStatusChange(user.id, "active", user.name)
                            }
                            className="px-3.5 py-1.5 rounded-lg border-[1.5px] border-sky-200 bg-sky-50 text-sky-700 text-xs font-semibold hover:bg-sky-100 transition-colors cursor-pointer"
                          >
                            ▶ Activate
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteUser(user.id, user.name)}
                          className="px-3.5 py-1.5 rounded-lg border-[1.5px] border-red-200 bg-red-50 text-red-500 text-xs font-semibold hover:bg-red-100 transition-colors cursor-pointer"
                        >
                          🗑 Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── LOGS ── */}
          {activeTab === "logs" && (
            <div className="fade-up">
              <div className="mb-6">
                <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                  Audit Logs
                </h1>
                <p className="text-sm text-slate-500 mt-1">
                  {filteredLogs.length} entries
                </p>
              </div>

              {/* Filter pills */}
              <div className="flex gap-2 mb-4 flex-wrap">
                {uniqueActions.map((a) => (
                  <button
                    key={a}
                    onClick={() => setLogFilter(a)}
                    className={`px-3.5 py-1.5 rounded-full text-xs font-semibold border-[1.5px] cursor-pointer transition-colors ${
                      logFilter === a
                        ? "border-indigo-500 bg-indigo-500 text-white"
                        : "border-slate-200 bg-white text-slate-500 hover:bg-slate-50"
                    }`}
                  >
                    {a}
                  </button>
                ))}
              </div>

              <div className="bg-white rounded-2xl overflow-hidden shadow-sm border border-slate-100">
                {filteredLogs.length === 0 ? (
                  <div className="py-16 px-8 text-center">
                    <div className="text-4xl mb-3">📋</div>
                    <p className="text-[15px] text-slate-500">
                      No logs for this filter
                    </p>
                  </div>
                ) : (
                  filteredLogs.map((log, i) => (
                    <div
                      key={log.id}
                      className={`grid grid-cols-[140px_auto_1fr_90px] px-5 py-3.5 items-center gap-3 transition-colors hover:bg-slate-50 ${i < filteredLogs.length - 1 ? "border-b border-slate-50" : ""}`}
                    >
                      <span className="text-[11px] font-mono text-slate-400">
                        {log.time?.replace("T", " ").slice(0, 19)}
                      </span>
                      <ActionBadge action={log.action} />
                      <div>
                        <span className="text-[13px] text-slate-700">
                          {log.description}
                        </span>
                        <span className="text-[11px] text-slate-400 ml-2">
                          by {log.actor}
                        </span>
                      </div>
                      <StatusBadge status={log.status} />
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {/* ── Add User Modal ── */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/40 z-[7000] flex items-center justify-center">
          <div className="bg-white rounded-[18px] p-8 w-[420px] shadow-2xl animate-[fadeUp_0.25s_ease]">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-bold text-slate-900">Add New User</h2>
              <button
                onClick={() => setShowAddModal(false)}
                className="bg-transparent border-none cursor-pointer text-xl text-slate-400 hover:text-slate-600"
              >
                ×
              </button>
            </div>

            <form
              onSubmit={handleAddEmployee}
              className="flex flex-col gap-3.5"
            >
              {[
                {
                  label: "Full Name",
                  key: "name",
                  type: "text",
                  placeholder: "John Doe",
                },
                {
                  label: "Email",
                  key: "email",
                  type: "email",
                  placeholder: "john@company.com",
                },
                {
                  label: "Password",
                  key: "password",
                  type: "password",
                  placeholder: "••••••••",
                },
              ].map(({ label, key, type, placeholder }) => (
                <div key={key}>
                  <label className="text-xs font-semibold text-slate-700 tracking-wide block mb-1.5">
                    {label.toUpperCase()}
                  </label>
                  <input
                    type={type}
                    placeholder={placeholder}
                    value={newEmp[key]}
                    onChange={(e) =>
                      setNewEmp({ ...newEmp, [key]: e.target.value })
                    }
                    className="w-full px-3.5 py-2.5 rounded-xl border-[1.5px] border-slate-200 text-[14px] text-slate-800 bg-slate-50 focus:outline-none focus:border-indigo-500 focus:bg-white transition-colors"
                  />
                </div>
              ))}

              <div className="flex gap-2.5 mt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="flex-1 py-2.5 rounded-xl border-[1.5px] border-slate-200 bg-white cursor-pointer text-sm text-slate-500 font-medium hover:bg-slate-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={addingUser}
                  className="flex-1 py-2.5 rounded-xl border-none bg-slate-900 text-white cursor-pointer text-sm font-semibold hover:bg-slate-800 hover:-translate-y-px transition-all shadow-sm disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {addingUser ? "Creating..." : "Create User"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
      {/* ── Duplicate File Modal ── */}
      {duplicateFile && (
        <div className="fixed inset-0 bg-black/40 z-[7000] flex items-center justify-center">
          <div className="bg-white rounded-[18px] p-8 w-[420px] shadow-2xl animate-[fadeUp_0.25s_ease]">
            <div className="text-3xl mb-4 text-center">⚠️</div>
            <h2 className="text-lg font-bold text-slate-900 mb-2 text-center">
              File Already Exists
            </h2>
            <p className="text-sm text-slate-500 text-center mb-6 leading-relaxed">
              <span className="font-semibold text-slate-700">
                "{duplicateFile.name}"
              </span>{" "}
              is already indexed. Overwrite it or cancel?
            </p>
            <div className="flex gap-2.5">
              <button
                onClick={() => {
                  setDuplicateFile(null);
                  setFiles([]);
                }}
                className="flex-1 py-2.5 rounded-xl border-[1.5px] border-slate-200 bg-white cursor-pointer text-sm text-slate-500 font-medium hover:bg-slate-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleOverwrite}
                disabled={uploading}
                className="flex-1 py-2.5 rounded-xl border-none bg-red-500 text-white cursor-pointer text-sm font-semibold hover:bg-red-600 transition-all shadow-sm disabled:opacity-70"
              >
                {uploading ? "Overwriting..." : "⚠ Overwrite"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

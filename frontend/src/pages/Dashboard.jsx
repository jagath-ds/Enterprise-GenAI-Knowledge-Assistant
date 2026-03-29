import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { queryAI, changePassword } from "../services/api";
import ReactMarkdown from "react-markdown";
import {
  LogOut,
  Plus,
  MessageSquare,
  Send,
  ChevronDown,
  MoreVertical,
  FileText,
  Pin,
  Trash2,
  Menu,
  Settings,
  Bell,
  Mail,
  Sun,
  Moon,
  Lock,
  X,
  Sparkles,
  Check,
  Copy,
  CheckCheck,
  FileIcon,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Premium Markdown Wrapper
// ---------------------------------------------------------------------------
function MarkdownMessage({ text, isDark }) {
  return (
    <ReactMarkdown
      components={{
        p: ({ children }) => (
          <p className="mb-4 last:mb-0 leading-relaxed text-[15px]">
            {children}
          </p>
        ),
        code: ({ inline, children }) =>
          inline ? (
            <code
              className={`px-1.5 py-0.5 rounded-md text-[13px] font-mono border ${
                isDark
                  ? "bg-white/10 border-white/10 text-blue-300"
                  : "bg-gray-100 border-gray-200 text-blue-600"
              }`}
            >
              {children}
            </code>
          ) : (
            <pre
              className={`my-4 p-4 rounded-xl text-[13px] font-mono overflow-x-auto border shadow-sm ${
                isDark
                  ? "bg-[#09090b] border-white/10 text-gray-300"
                  : "bg-gray-50 border-gray-200 text-gray-800"
              }`}
            >
              <code>{children}</code>
            </pre>
          ),
        ul: ({ children }) => (
          <ul className="list-disc pl-5 mb-4 space-y-2 marker:text-gray-400">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal pl-5 mb-4 space-y-2 marker:text-gray-400">
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li className="pl-1 text-[15px] leading-relaxed">{children}</li>
        ),
        strong: ({ children }) => (
          <strong
            className={`font-semibold ${isDark ? "text-gray-100" : "text-gray-900"}`}
          >
            {children}
          </strong>
        ),
        h1: ({ children }) => (
          <h1 className="text-xl font-bold mb-4 mt-6 first:mt-0 tracking-tight">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-lg font-bold mb-3 mt-5 first:mt-0 tracking-tight">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-base font-semibold mb-2 mt-4 tracking-tight">
            {children}
          </h3>
        ),
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

// ---------------------------------------------------------------------------
// Smooth Copy Button
// ---------------------------------------------------------------------------
function CopyButton({ text, isDark }) {
  const [copied, setCopied] = useState(false);
  const handle = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handle}
      title="Copy"
      className={`p-1.5 rounded-lg transition-all duration-200 opacity-0 group-hover:opacity-100 ${
        isDark
          ? "hover:bg-white/10 text-gray-500 hover:text-gray-300"
          : "hover:bg-gray-100 text-gray-400 hover:text-gray-700"
      }`}
    >
      {copied ? (
        <CheckCheck size={16} className="text-green-500" />
      ) : (
        <Copy size={16} />
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard
// ---------------------------------------------------------------------------
export default function Dashboard() {
  const navigate = useNavigate();
  const [input, setInput] = useState("");
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [currentModel, setCurrentModel] = useState("auto");
  const [modelMode, setModelMode] = useState("auto");
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isTyping, setIsTyping] = useState(false);
  const [theme, setTheme] = useState(
    localStorage.getItem("app-theme") || "light",
  );
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [passMsg, setPassMsg] = useState("");
  const [passwordInput, setPasswordInput] = useState("");
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [activeChatId, setActiveChatId] = useState(null);

  const messagesEndRef = useRef(null);
  const modelDropdownRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  const isDark = theme === "dark";

  // Persist theme
  useEffect(() => {
    localStorage.setItem("app-theme", theme);
  }, [theme]);

  // Auth guard
  useEffect(() => {
    const token = localStorage.getItem("token");
    const role = localStorage.getItem("role");
    if (!token || role !== "employee") navigate("/");
  }, [navigate]);

  // Close model dropdown
  useEffect(() => {
    const handler = (e) => {
      if (
        modelDropdownRef.current &&
        !modelDropdownRef.current.contains(e.target)
      )
        setIsModelDropdownOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Persist chats
  const [savedChats, setSavedChats] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("saved-chats") || "[]");
    } catch {
      return [];
    }
  });

  useEffect(() => {
    localStorage.setItem("saved-chats", JSON.stringify(savedChats));
  }, [savedChats]);

  const [messages, setMessages] = useState([]);

  useEffect(() => {
    if (activeChatId) {
      const chat = savedChats.find((c) => c.id === activeChatId);
      if (chat) setMessages(chat.messages);
    } else {
      setMessages([]);
    }
  }, [activeChatId, savedChats]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Auto-grow textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  }, [input]);

  const handleLogout = async () => {
    try {
      await fetch("http://localhost:8000/auth/logout", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify({
          user_id: localStorage.getItem("user_id"),
          role: localStorage.getItem("role"),
        }),
      });
    } catch (err) {
      console.error("Logout error:", err);
    }
    localStorage.clear();
    navigate("/");
  };

  const handleNewChat = () => {
    setActiveChatId(null);
    setMessages([]);
    setInput("");
    setSelectedFiles([]);
    setIsMenuOpen(false);
  };

  const handleDeleteChat = () => {
    if (activeChatId) {
      if (window.confirm("Delete this chat?")) {
        setSavedChats((prev) => prev.filter((c) => c.id !== activeChatId));
        handleNewChat();
      }
    } else {
      setMessages([]);
    }
    setIsMenuOpen(false);
  };

  const handleFileChange = (e) => {
    if (e.target.files)
      setSelectedFiles((prev) => [...prev, ...Array.from(e.target.files)]);
  };

  const removeFile = (idx) =>
    setSelectedFiles((prev) => prev.filter((_, i) => i !== idx));

  const handleSend = async () => {
    if (isTyping) return;
    if (!input.trim() && selectedFiles.length === 0) return;

    const userInput = input;
    const newUserMessage = {
      role: "user",
      text: userInput,
      files: selectedFiles,
    };
    let currentChatId = activeChatId;

    if (!currentChatId) {
      const newId = Date.now();
      const newTitle =
        userInput.length > 30 ? userInput.substring(0, 30) + "…" : userInput;
      const newChat = {
        id: newId,
        title: newTitle,
        time: "Just now",
        messages: [newUserMessage],
      };
      setSavedChats((prev) => [newChat, ...prev]);
      setActiveChatId(newId);
      currentChatId = newId;
    } else {
      setSavedChats((prev) =>
        prev.map((c) =>
          c.id === currentChatId
            ? { ...c, messages: [...c.messages, newUserMessage] }
            : c,
        ),
      );
    }

    setMessages((prev) => [...prev, newUserMessage]);
    setInput("");
    setSelectedFiles([]);
    setIsTyping(true);

    try {
      const data = await queryAI(userInput, modelMode);
      const aiResponse = {
        role: "ai",
        text: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, aiResponse]);
      setSavedChats((prev) =>
        prev.map((c) =>
          c.id === currentChatId
            ? { ...c, messages: [...c.messages, aiResponse] }
            : c,
        ),
      );
    } catch {
      const err = {
        role: "ai",
        text: "⚠️ Error contacting AI server. Please try again.",
      };
      setMessages((prev) => [...prev, err]);
    } finally {
      setIsTyping(false);
    }
  };

  const userName = localStorage.getItem("user_name") || "Employee";
  const userInitial = userName.charAt(0).toUpperCase();

  return (
    <div
      className={`flex h-screen font-sans overflow-hidden transition-colors duration-300 ${isDark ? "bg-[#09090b] text-gray-200" : "bg-white text-gray-800"}`}
    >
      {/* ── Sidebar ── */}
      <aside
        className={`
        ${isDark ? "bg-[#09090b] border-white/10" : "bg-[#fcfcfc] border-gray-200"}
        border-r flex flex-col transition-all duration-300 ease-in-out shrink-0
        ${isSidebarOpen ? "w-64" : "w-0 border-none"} overflow-hidden whitespace-nowrap
      `}
      >
        <div className="p-5 flex items-center gap-3 mb-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center font-bold text-white text-lg shadow-sm">
            S
          </div>
          <span
            className={`font-bold tracking-tight text-lg ${isDark ? "text-gray-100" : "text-gray-900"}`}
          >
            Strokx
          </span>
        </div>

        <button
          onClick={handleNewChat}
          className={`mx-4 flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all mb-6 group ${
            isDark
              ? "bg-white/5 hover:bg-white/10 text-gray-200"
              : "bg-white border border-gray-200 hover:border-blue-200 hover:bg-blue-50/50 text-gray-700 hover:text-blue-700 shadow-sm"
          }`}
        >
          <Plus
            size={18}
            className="shrink-0 transition-transform group-hover:rotate-90"
          />
          New Chat
        </button>

        <div className="flex-grow px-3 overflow-y-auto [&::-webkit-scrollbar]:hidden">
          <div className="px-2 text-[11px] font-bold text-gray-400 mb-3 uppercase tracking-wider">
            Recent Chats
          </div>

          {savedChats.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center px-4">
              <MessageSquare
                size={24}
                className={isDark ? "text-gray-700 mb-3" : "text-gray-300 mb-3"}
              />
              <p className="text-sm font-medium text-gray-500">No chats yet</p>
            </div>
          ) : (
            <div className="space-y-0.5">
              {savedChats.map((chat) => (
                <div
                  key={chat.id}
                  onClick={() => {
                    setActiveChatId(chat.id);
                    setMessages(chat.messages);
                  }}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer text-sm font-medium transition-colors ${
                    activeChatId === chat.id
                      ? isDark
                        ? "bg-white/10 text-white"
                        : "bg-blue-50 text-blue-700"
                      : isDark
                        ? "text-gray-400 hover:text-gray-200 hover:bg-white/5"
                        : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                  }`}
                >
                  <MessageSquare size={16} className="shrink-0 opacity-70" />
                  <span className="truncate">{chat.title}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* User Footer */}
        <div
          className={`p-4 border-t ${isDark ? "border-white/10" : "border-gray-200"}`}
        >
          <div className="flex items-center gap-3 px-2 mb-4">
            <div
              className={`w-9 h-9 rounded-full flex items-center justify-center font-bold text-sm shrink-0 ${
                isDark
                  ? "bg-indigo-500/20 text-indigo-300"
                  : "bg-indigo-100 text-indigo-700"
              }`}
            >
              {userInitial}
            </div>
            <div className="flex flex-col min-w-0">
              <span
                className={`font-semibold text-sm truncate ${isDark ? "text-gray-100" : "text-gray-900"}`}
              >
                {userName}
              </span>
              <span className="text-[11px] text-gray-500 font-medium">
                Employee
              </span>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className={`w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition text-sm font-medium ${
              isDark
                ? "text-gray-400 hover:text-white hover:bg-white/5"
                : "text-gray-600 hover:text-red-600 hover:bg-red-50"
            }`}
          >
            <LogOut size={16} /> Log out
          </button>
        </div>
      </aside>

      {/* ── Main Canvas ── */}
      <main
        className={`flex-grow flex flex-col relative transition-colors duration-300 ${isDark ? "bg-[#09090b]" : "bg-white"}`}
      >
        {/* Top Header */}
        <header
          className={`w-full px-5 py-3 flex justify-between items-center z-20 ${isDark ? "bg-[#09090b]" : "bg-white/80 backdrop-blur-md border-b border-gray-100"}`}
        >
          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className={`p-2 rounded-lg transition ${isDark ? "text-gray-400 hover:text-white hover:bg-white/10" : "text-gray-500 hover:bg-gray-100"}`}
            >
              <Menu size={20} />
            </button>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setIsSettingsOpen(true)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition ${
                isDark
                  ? "text-gray-400 hover:text-white hover:bg-white/10"
                  : "text-gray-600 hover:bg-gray-100"
              }`}
            >
              <Settings size={16} />{" "}
              <span className="hidden md:inline">Settings</span>
            </button>

            <div
              className={`w-px h-5 mx-2 ${isDark ? "bg-white/10" : "bg-gray-200"}`}
            />

            <div className="relative">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className={`p-2 rounded-lg transition ${isDark ? "text-gray-400 hover:text-white hover:bg-white/10" : "text-gray-500 hover:bg-gray-100"}`}
              >
                <MoreVertical size={20} />
              </button>
              {isMenuOpen && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setIsMenuOpen(false)}
                  />
                  <div
                    className={`absolute right-0 top-full mt-2 w-48 rounded-xl shadow-lg z-20 overflow-hidden py-1 border ${
                      isDark
                        ? "bg-[#18181b] border-white/10"
                        : "bg-white border-gray-100"
                    }`}
                  >
                    <button
                      className={`w-full text-left px-4 py-2.5 text-sm font-medium flex items-center gap-3 transition ${isDark ? "hover:bg-white/5 text-gray-200" : "hover:bg-gray-50 text-gray-700"}`}
                    >
                      <FileText size={16} /> Files
                    </button>
                    <button
                      className={`w-full text-left px-4 py-2.5 text-sm font-medium flex items-center gap-3 transition ${isDark ? "hover:bg-white/5 text-gray-200" : "hover:bg-gray-50 text-gray-700"}`}
                    >
                      <Pin size={16} /> Pin Chat
                    </button>
                    <div
                      className={`h-px my-1 mx-3 ${isDark ? "bg-white/10" : "bg-gray-100"}`}
                    />
                    <button
                      onClick={handleDeleteChat}
                      className={`w-full text-left px-4 py-2.5 text-sm font-medium text-red-500 flex items-center gap-3 transition ${isDark ? "hover:bg-red-500/10" : "hover:bg-red-50"}`}
                    >
                      <Trash2 size={16} /> Delete Chat
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </header>

        {/* ── Chat Area ── */}
        <section className="flex-grow overflow-y-auto px-4 py-8 [&::-webkit-scrollbar]:hidden scroll-smooth">
          {messages.length === 0 ? (
            /* Premium Welcome Screen */
            <div className="h-full flex flex-col items-center justify-center -mt-12 animate-in fade-in duration-700">
              <div className="relative mb-8">
                {isDark && (
                  <div className="absolute inset-0 blur-[80px] opacity-30 rounded-full w-40 h-40 mx-auto bg-indigo-500" />
                )}
                <div className="relative w-16 h-16 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-xl">
                  <Sparkles size={32} />
                </div>
              </div>
              <h2
                className={`text-3xl font-bold tracking-tight mb-3 ${isDark ? "text-white" : "text-gray-900"}`}
              >
                How can I help you today?
              </h2>
              <p
                className={`text-base mb-10 ${isDark ? "text-gray-400" : "text-gray-500"}`}
              >
                Ask me anything about your company documents.
              </p>

              <div className="flex flex-wrap gap-3 justify-center max-w-2xl">
                {[
                  "Summarize the SOP manual",
                  "What is the leave policy?",
                  "Explain the onboarding process",
                  "Who do I contact for IT support?",
                ].map((s) => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className={`px-5 py-2.5 rounded-full text-[14px] font-medium transition-all duration-200 border ${
                      isDark
                        ? "border-white/10 text-gray-300 bg-white/5 hover:border-indigo-500/50 hover:bg-indigo-500/10"
                        : "border-gray-200 text-gray-600 bg-white hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700 shadow-sm"
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-8 pb-10">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex gap-4 fade-up ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {/* AI Avatar */}
                  {msg.role === "ai" && (
                    <div className="relative mt-1 shrink-0 h-8">
                      {isDark && (
                        <div className="absolute inset-0 blur-lg bg-indigo-500 opacity-40 rounded-full"></div>
                      )}
                      <div className="relative w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md">
                        <Sparkles size={14} />
                      </div>
                    </div>
                  )}

                  <div
                    className={`group flex flex-col gap-1.5 ${msg.role === "user" ? "items-end max-w-[80%]" : "items-start w-full min-w-0"}`}
                  >
                    {/* User Bubble */}
                    {msg.role === "user" ? (
                      <div className="px-5 py-3.5 bg-blue-600 text-white rounded-2xl rounded-br-sm shadow-md text-[15px] leading-relaxed">
                        {msg.files && msg.files.length > 0 && (
                          <div className="mb-2 flex flex-wrap gap-1.5">
                            {msg.files.map((f, fi) => (
                              <span
                                key={fi}
                                className="flex items-center gap-1.5 text-xs bg-black/20 px-2.5 py-1 rounded-md font-medium"
                              >
                                <FileIcon size={12} /> {f}
                              </span>
                            ))}
                          </div>
                        )}
                        <p className="whitespace-pre-wrap break-words">
                          {msg.text}
                        </p>
                      </div>
                    ) : (
                      /* AI Text Block (No containing box) */
                      <div className="w-full pr-4">
                        <MarkdownMessage text={msg.text} isDark={isDark} />

                        {/* Sources */}
                        {msg.sources && msg.sources.length > 0 && (
                          <div
                            className={`mt-5 pt-4 border-t ${isDark ? "border-white/10" : "border-gray-200"}`}
                          >
                            <p
                              className={`text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-1.5 ${isDark ? "text-gray-500" : "text-gray-400"}`}
                            >
                              <FileText size={14} /> References
                            </p>
                            <div className="flex flex-wrap gap-2">
                              {[
                                ...new Set(
                                  msg.sources.map((s) => s.source_path),
                                ),
                              ].map((file, idx) => {
                                const pages = msg.sources
                                  .filter((s) => s.source_path === file)
                                  .map((s) => s.page)
                                  .join(", ");
                                return (
                                  <div
                                    key={idx}
                                    className={`flex items-start gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border ${
                                      isDark
                                        ? "bg-white/5 border-white/10 text-gray-300"
                                        : "bg-gray-50 border-gray-200 text-gray-600"
                                    }`}
                                  >
                                    {/* Removed truncation, added break-words */}
                                    <span className="break-words">{file}</span>

                                    {/* Added shrink-0 so the page numbers don't get squished by long file names */}
                                    <span
                                      className={`shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-[10px] ${isDark ? "bg-black/50 text-gray-400" : "bg-white border text-gray-500"}`}
                                    >
                                      p. {pages}
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* AI Copy Button */}
                    {msg.role === "ai" && (
                      <div className="mt-1">
                        <CopyButton text={msg.text} isDark={isDark} />
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Typing Indicator */}
              {isTyping && (
                <div className="flex gap-4 justify-start fade-up">
                  <div className="relative mt-1 shrink-0 h-8">
                    {isDark && (
                      <div className="absolute inset-0 blur-lg bg-indigo-500 opacity-40 rounded-full"></div>
                    )}
                    <div className="relative w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md">
                      <Sparkles size={14} />
                    </div>
                  </div>
                  <div className="flex items-center h-10 px-4">
                    <div className="flex gap-1.5 items-center">
                      <span
                        className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      />
                      <span
                        className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      />
                      <span
                        className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </section>

        {/* ── Floating Input Dock ── */}
        <div
          className={`px-4 pb-6 pt-2 w-full z-20 ${isDark ? "bg-gradient-to-t from-[#09090b] via-[#09090b] to-transparent" : "bg-gradient-to-t from-white via-white to-transparent"}`}
        >
          <div className="max-w-3xl mx-auto">
            {/* Selected Files Preview */}
            {selectedFiles.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-3 px-2">
                {selectedFiles.map((f, i) => (
                  <div
                    key={i}
                    className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full font-medium border ${
                      isDark
                        ? "bg-zinc-800/80 border-white/10 text-gray-200"
                        : "bg-white border-gray-200 text-gray-700 shadow-sm"
                    }`}
                  >
                    <FileIcon
                      size={12}
                      className={isDark ? "text-indigo-400" : "text-indigo-600"}
                    />
                    <span className="max-w-[150px] truncate">{f.name}</span>
                    <button
                      onClick={() => removeFile(i)}
                      className="hover:text-red-500 transition ml-1"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Input Bar */}
            <div
              className={`relative flex items-end gap-2 border rounded-3xl p-2 transition-all duration-300 shadow-sm ${
                isDark
                  ? "bg-[#18181b] border-white/10 focus-within:border-indigo-500/50 focus-within:bg-[#18181b] focus-within:ring-4 focus-within:ring-indigo-500/10"
                  : "bg-gray-50 border-gray-200 focus-within:border-indigo-400 focus-within:bg-white focus-within:ring-4 focus-within:ring-indigo-500/10 focus-within:shadow-md"
              }`}
            >
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`p-2.5 rounded-full transition shrink-0 mb-0.5 ${isDark ? "text-gray-400 hover:text-white hover:bg-white/10" : "text-gray-500 hover:text-gray-800 hover:bg-gray-200"}`}
                title="Attach file"
              >
                <Plus size={20} />
              </button>
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                multiple
                onChange={handleFileChange}
              />

              <textarea
                ref={textareaRef}
                rows={1}
                className={`flex-grow bg-transparent py-3 outline-none text-[15px] resize-none leading-relaxed max-h-40 ${
                  isDark
                    ? "text-gray-100 placeholder-gray-500"
                    : "text-gray-900 placeholder-gray-400"
                }`}
                placeholder="Ask about your company documents..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !isTyping) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
              />

              <button
                onClick={handleSend}
                disabled={
                  (!input.trim() && selectedFiles.length === 0) || isTyping
                }
                className={`p-2.5 rounded-full transition-all duration-200 shrink-0 mb-0.5 flex items-center justify-center ${
                  input.trim() || selectedFiles.length > 0
                    ? "bg-blue-600 text-white shadow-md hover:bg-blue-700 hover:scale-105 active:scale-95"
                    : isDark
                      ? "bg-white/5 text-gray-600 cursor-not-allowed"
                      : "bg-gray-200 text-gray-400 cursor-not-allowed"
                }`}
              >
                <Send
                  size={18}
                  className={
                    input.trim() || selectedFiles.length > 0
                      ? "translate-x-0.5"
                      : ""
                  }
                />
              </button>
            </div>

            {/* Bottom Footer Area */}
            <div className="mt-3 flex items-center justify-between px-2">
              <div className="flex items-center gap-2" ref={modelDropdownRef}>
                <div className="relative">
                  <button
                    onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition border ${
                      isDark
                        ? "bg-[#18181b] border-white/10 text-gray-400 hover:text-gray-200"
                        : "bg-white border-gray-200 text-gray-500 hover:text-gray-700 shadow-sm"
                    }`}
                  >
                    <Sparkles
                      size={12}
                      className={isDark ? "text-indigo-400" : "text-indigo-500"}
                    />
                    {currentModel}
                    <ChevronDown
                      size={12}
                      className={`${isModelDropdownOpen ? "rotate-180" : ""} transition-transform opacity-70`}
                    />
                  </button>
                  {isModelDropdownOpen && (
                    <div
                      className={`absolute bottom-full left-0 mb-2 w-48 border rounded-xl shadow-xl overflow-hidden z-50 py-1.5 ${
                        isDark
                          ? "bg-[#18181b] border-white/10"
                          : "bg-white border-gray-100"
                      }`}
                    >
                      {[
                        { label: "Auto", value: "auto" },
                        { label: "⚡ Fast", value: "fast" },
                        { label: "🧠 Deep", value: "deep" },
                      ].map((m) => (
                        <button
                          key={m.value}
                          onClick={() => {
                            setCurrentModel(m.label);
                            setModelMode(m.value); // ⭐ IMPORTANT
                            setIsModelDropdownOpen(false);
                          }}
                          className={`w-full text-left px-4 py-2.5 text-xs font-medium flex items-center justify-between transition ${
                            isDark
                              ? "hover:bg-white/5 text-gray-300"
                              : "hover:bg-gray-50 text-gray-700"
                          }`}
                        >
                          {m.label}
                          {currentModel === m.label && (
                            <Check size={14} className="text-indigo-500" />
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <span
                className={`text-[11px] font-medium ${isDark ? "text-gray-600" : "text-gray-400"}`}
              >
                Shift + Enter for new line
              </span>
            </div>
          </div>
        </div>

        {/* ── Premium Settings Modal ── */}
        {isSettingsOpen && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
            <div
              className={`w-full max-w-sm rounded-2xl border shadow-2xl overflow-hidden scale-in-95 animate-in duration-200 ${
                isDark
                  ? "bg-[#09090b] border-white/10"
                  : "bg-white border-gray-100"
              }`}
            >
              <div
                className={`px-6 py-4 border-b flex justify-between items-center ${isDark ? "border-white/10" : "border-gray-100"}`}
              >
                <h2
                  className={`font-semibold text-lg flex items-center gap-2 ${isDark ? "text-white" : "text-gray-900"}`}
                >
                  <Settings size={18} /> Settings
                </h2>
                <button
                  onClick={() => setIsSettingsOpen(false)}
                  className={`p-1.5 rounded-lg transition ${isDark ? "hover:bg-white/10 text-gray-400 hover:text-white" : "hover:bg-gray-100 text-gray-500 hover:text-gray-900"}`}
                >
                  <X size={20} />
                </button>
              </div>

              <div className="p-6 space-y-8">
                {/* Appearance */}
                <div className="flex items-center justify-between">
                  <div>
                    <p
                      className={`font-semibold text-sm ${isDark ? "text-gray-100" : "text-gray-900"}`}
                    >
                      Appearance
                    </p>
                    <p
                      className={`text-xs mt-1 ${isDark ? "text-gray-500" : "text-gray-500"}`}
                    >
                      Toggle app theme
                    </p>
                  </div>
                  <button
                    onClick={() => setTheme(isDark ? "light" : "dark")}
                    className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition border ${
                      isDark
                        ? "bg-[#18181b] border-white/10 text-gray-300 hover:bg-white/5"
                        : "bg-gray-50 border-gray-200 text-gray-700 hover:bg-gray-100 shadow-sm"
                    }`}
                  >
                    {isDark ? (
                      <>
                        <Sun size={16} /> Light
                      </>
                    ) : (
                      <>
                        <Moon size={16} /> Dark
                      </>
                    )}
                  </button>
                </div>

                {/* Security */}
                <div
                  className={`pt-6 border-t space-y-4 ${isDark ? "border-white/10" : "border-gray-100"}`}
                >
                  <p
                    className={`font-semibold text-sm flex items-center gap-2 ${isDark ? "text-gray-100" : "text-gray-900"}`}
                  >
                    <Lock size={16} /> Security
                  </p>
                  <input
                    type="password"
                    placeholder="New password"
                    value={passwordInput}
                    onChange={(e) => setPasswordInput(e.target.value)}
                    className={`w-full px-4 py-3 rounded-xl border text-sm outline-none transition-all duration-200 ${
                      isDark
                        ? "bg-[#18181b] border-white/10 text-gray-200 placeholder-gray-600 focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/20"
                        : "bg-gray-50 border-gray-200 text-gray-900 placeholder-gray-400 focus:border-indigo-400 focus:bg-white focus:ring-2 focus:ring-indigo-500/10"
                    }`}
                  />
                  <button
                    onClick={async () => {
                      try {
                        await changePassword(passwordInput);
                        setPassMsg("Password Updated Successfully");
                      } catch (err) {
                        setPassMsg("Error updating password");
                      }
                      setTimeout(() => setPassMsg(""), 3000);
                    }}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm font-semibold py-3 rounded-xl transition shadow-md hover:shadow-lg"
                  >
                    Update Password
                  </button>
                  {passMsg && (
                    <p
                      className={`text-center text-xs font-semibold flex items-center justify-center gap-1.5 ${passMsg.includes("Error") ? "text-red-500" : "text-green-500"}`}
                    >
                      {passMsg.includes("Error") ? (
                        <X size={14} />
                      ) : (
                        <Check size={14} />
                      )}{" "}
                      {passMsg}
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

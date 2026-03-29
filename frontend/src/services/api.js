
const BASE_URL = 'http://localhost:8000';

export const queryAI = async (question, modelMode = "auto") => {
    try {
        const response = await fetch(`${BASE_URL}/api/query/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                model_mode: modelMode   // ⭐ IMPORTANT
            }),
        });

        if (!response.ok) throw new Error('Failed to fetch response');
        return await response.json();
    } catch (error) {
        console.error("API Error:", error);
        return { answer: "Error connecting to the server." };
    }
};

export const uploadDocuments = async (files) => {
    for (let file of files) {
        const formData = new FormData();
        formData.append("file", file);
        await fetch(`${BASE_URL}/api/documents/upload`, {
            method: "POST",
            body: formData,
        });
    }
};

export const getDocuments = async () => {
    const res = await fetch(`${BASE_URL}/api/documents/`);
    return await res.json();
};

export const deleteDocument = async (id) => {
    await fetch(`${BASE_URL}/api/documents/${id}`, { method: "DELETE" });
};

export const reindexDocument = async (id) => {
    const res = await fetch(`${BASE_URL}/api/documents/${id}/index?force=true`, { 
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({}) 
    });

    if (!res.ok) {
        const errorData = await res.json().catch(() => null);
        console.error("Backend Error Details:", errorData);
        throw new Error("Failed to reindex document");
    }
    
    return res.json().catch(() => ({}));
};

export const employeeLogin = async (email, password) => {
    const res = await fetch(`${BASE_URL}/auth/employee/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
    });
    return res.json();
};

export const adminLogin = async (email, password) => {
    const res = await fetch(`${BASE_URL}/auth/admin/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
    });
    return res.json();
};

export const getEmployees = async () => {
    const token = localStorage.getItem("token");
    const res = await fetch(`${BASE_URL}/admin/employees`, {
        headers: { Authorization: `Bearer ${token}` },
    });
    return res.json();
};

export const approveUser = async (id) => {
    const token = localStorage.getItem("token");
    await fetch(`${BASE_URL}/auth/admin/approve/${id}`, {
        method: "PATCH",
        headers: { Authorization: `Bearer ${token}` },
    });
};

export const registerUser = async ({ firstname, lastname, email, password }) => {
    const res = await fetch(`${BASE_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ firstname, lastname, email, password }),
    });
    return res.json();
};

export const updateUserStatus = async (id, status) => {
    const token = localStorage.getItem("token");
    await fetch(`${BASE_URL}/admin/employees/${id}/status?status=${status}`, {
        method: "PATCH",
        headers: { Authorization: `Bearer ${token}` },
    });
};

export const deleteUser = async (id) => {
    const token = localStorage.getItem("token");
    await fetch(`${BASE_URL}/admin/employees/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
    });
};

export const getLogs = async () => {
    const token = localStorage.getItem("token");
    const res = await fetch(`${BASE_URL}/admin/logs`, {
        headers: { Authorization: `Bearer ${token}` },
    });
    return res.json();
};

export const changePassword = async (newPassword) => {
    const res = await fetch(`${BASE_URL}/auth/change-password`, {
        method: "PUT",
        headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("token")}`,
        },
        body: JSON.stringify({ new_password: newPassword }),
    });
    return res.json();
};
import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import socket

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "sessions")

# HTML Template with beautiful glassmorphism, dark/light theme, search, auto-refresh and print features.
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Acadza Markdown Live Viewer</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- KaTeX for high-performance math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    
    <!-- Marked for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    
    <!-- Prism JS for code syntax highlighting -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css">
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
    
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
        :root {
            --bg-color: #0b0f19;
            --sidebar-bg: #111827;
            --card-bg: rgba(22, 28, 45, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-color: #4f46e5;
            --accent-gradient: linear-gradient(135deg, #6366f1, #3b82f6);
            --hover-bg: rgba(255, 255, 255, 0.05);
            --active-bg: rgba(99, 102, 241, 0.2);
            --active-border: #6366f1;
            --sidebar-width: 320px;
            --header-height: 70px;
        }

        .light-theme {
            --bg-color: #ffffff;
            --sidebar-bg: #ffffff;
            --card-bg: #ffffff;
            --border-color: rgba(0, 0, 0, 0.08);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --accent-color: #3b82f6;
            --accent-gradient: linear-gradient(135deg, #3b82f6, #60a5fa);
            --hover-bg: rgba(0, 0, 0, 0.04);
            --active-bg: rgba(59, 130, 246, 0.15);
            --active-border: #3b82f6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            display: flex;
            height: 100vh;
            overflow: hidden;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        /* Sidebar styling */
        .sidebar {
            width: var(--sidebar-width);
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            height: 100%;
            flex-shrink: 0;
            transition: background-color 0.3s ease, border-color 0.3s ease;
            z-index: 10;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .sidebar-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .search-container {
            position: relative;
        }

        .search-input {
            width: 100%;
            padding: 10px 16px 10px 38px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-size: 0.875rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .search-input:focus {
            border-color: var(--active-border);
            box-shadow: 0 0 0 2px var(--active-bg);
        }

        .search-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        .file-list {
            flex-grow: 1;
            overflow-y: auto;
            padding: 15px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .file-item {
            padding: 12px 14px;
            border-radius: 8px;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.2s ease;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .file-item:hover {
            background-color: var(--hover-bg);
        }

        .file-item.active {
            background-color: var(--active-bg);
            border-color: var(--active-border);
        }

        .file-name {
            font-weight: 500;
            font-size: 0.9rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .file-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
        }

        /* Main Content styling */
        .main-container {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
            position: relative;
            overflow: hidden;
        }

        /* Header styling */
        .top-header {
            height: var(--header-height);
            border-bottom: 1px solid var(--border-color);
            padding: 0 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: var(--sidebar-bg);
            backdrop-filter: blur(12px);
            z-index: 5;
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }

        .active-file-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .active-file-title i {
            color: var(--accent-color);
        }

        .toolbar {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .btn {
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 8px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }

        .btn:hover {
            background-color: var(--hover-bg);
            border-color: var(--text-secondary);
        }

        .btn-primary {
            background: var(--accent-gradient);
            color: white;
            border: none;
        }

        .btn-primary:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }

        .refresh-toggle-container {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
            padding: 6px 12px;
            border-radius: 6px;
            background: var(--hover-bg);
        }

        .switch {
            position: relative;
            display: inline-block;
            width: 34px;
            height: 20px;
        }

        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .3s;
            border-radius: 20px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 14px;
            width: 14px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background-color: var(--accent-color);
        }

        input:checked + .slider:before {
            transform: translateX(14px);
        }

        /* Document Viewer styling */
        .viewer-container {
            flex-grow: 1;
            overflow-y: auto;
            padding: 40px 30px;
            display: flex;
            justify-content: center;
        }

        .document-card {
            width: 100%;
            max-width: 900px;
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 50px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            min-height: 100%;
            transition: background-color 0.3s, border-color 0.3s;
        }

        /* Rendered Markdown Styling */
        .markdown-body {
            font-family: 'Inter', sans-serif;
            line-height: 1.7;
            font-size: 1.05rem;
        }

        .markdown-body h1, 
        .markdown-body h2, 
        .markdown-body h3, 
        .markdown-body h4 {
            font-family: 'Outfit', sans-serif;
            color: var(--text-primary);
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 700;
        }

        .markdown-body h1 { font-size: 2rem; border-bottom: 2px solid var(--border-color); padding-bottom: 10px; margin-top: 0; }
        .markdown-body h2 { font-size: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; }
        .markdown-body h3 { font-size: 1.25rem; }

        .markdown-body p {
            margin-bottom: 1.2em;
            color: var(--text-primary);
        }

        /* Custom Table Styling */
        .markdown-body table {
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            font-size: 0.95rem;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
        }

        .markdown-body th {
            background: var(--accent-gradient);
            color: #ffffff;
            font-weight: 600;
            text-align: left;
            padding: 14px 18px;
            border: 1px solid var(--border-color);
        }

        .markdown-body td {
            padding: 14px 18px;
            border: 1px solid var(--border-color);
            background-color: rgba(255, 255, 255, 0.02);
            transition: background-color 0.2s;
        }

        .light-theme .markdown-body td {
            background-color: rgba(0, 0, 0, 0.01);
        }

        .markdown-body tr:hover td {
            background-color: var(--hover-bg);
        }

        /* Equation and Image Styling */
        .markdown-body img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 8px;
            padding: 8px;
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.3s;
        }

        .light-theme .markdown-body img {
            background-color: #ffffff;
        }

        .markdown-body img:hover {
            transform: scale(1.03);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        }

        /* Math and Latex Styling */
        .katex-display {
            overflow-x: auto;
            overflow-y: hidden;
            padding: 15px 0;
            margin: 1.5em 0;
            background-color: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .katex {
            font-size: 1.1em;
        }

        /* Code Blocks */
        .markdown-body pre {
            background-color: #0f1423 !important;
            padding: 18px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            margin: 20px 0;
            overflow-x: auto;
        }

        .markdown-body code {
            font-family: 'Fira Code', monospace;
            font-size: 0.9em;
            background-color: rgba(255, 255, 255, 0.08);
            padding: 2px 6px;
            border-radius: 4px;
        }

        .markdown-body pre code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
        }

        /* Empty State */
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-secondary);
            gap: 16px;
            text-align: center;
            padding: 40px;
        }

        .empty-state i {
            font-size: 4rem;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Print styles */
        @media print {
            body {
                background: white !important;
                color: black !important;
            }
            .sidebar, .top-header, .no-print {
                display: none !important;
            }
            .viewer-container {
                padding: 0 !important;
                overflow: visible !important;
                height: auto !important;
            }
            .document-card {
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
                max-width: 100% !important;
                background: white !important;
            }
            .markdown-body {
                color: black !important;
            }
            .markdown-body td, .markdown-body th {
                border: 1px solid #ddd !important;
                background-color: transparent !important;
                color: black !important;
            }
            .markdown-body th {
                background: #f0f0f0 !important;
            }
        }
        .html-sandbox {
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            height: 100%;
        }
        .html-sandbox textarea {
            width: 100%;
            height: 350px;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .html-sandbox textarea:focus {
            border-color: var(--active-border);
            box-shadow: 0 0 0 2px var(--active-bg);
        }
        .special-paste-btn {
            background-color: rgba(99, 102, 241, 0.08) !important;
            border: 1px dashed rgba(99, 102, 241, 0.3) !important;
            margin: 15px 15px 5px 15px !important;
            padding: 12px 14px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .special-paste-btn:hover {
            background-color: rgba(99, 102, 241, 0.15) !important;
            border-color: var(--accent-color) !important;
        }
        .special-paste-btn.active {
            background-color: rgba(99, 102, 241, 0.2) !important;
            border: 1px solid var(--accent-color) !important;
        }
    </style>
</head>
<body class="light-theme">

    <!-- Sidebar with PDF/MD files list -->
    <div class="sidebar no-print">
        <div class="sidebar-header">
            <div class="sidebar-title">
                <i class="fa-solid fa-file-invoice"></i> Acadza OCR Viewer
            </div>
            <div class="search-container">
                <i class="fa-solid fa-magnifying-glass search-icon"></i>
                <input type="text" id="search-input" class="search-input" placeholder="Search markdown files...">
            </div>
        </div>
        <div class="file-item special-paste-btn" id="paste-html-btn">
            <div class="file-name" style="color: #818cf8; font-weight: 600;"><i class="fa-solid fa-code"></i> Paste &amp; Render HTML</div>
        </div>
        <div class="file-list" id="file-list">
            <!-- Populated dynamically -->
        </div>
    </div>

    <!-- Main Rendering Pane -->
    <div class="main-container">
        <header class="top-header no-print">
            <div class="active-file-title" id="active-file-title">
                <i class="fa-solid fa-file-lines"></i> No Document Selected
            </div>
            <div class="toolbar">
                <div class="refresh-toggle-container">
                    <span>Auto Refresh</span>
                    <label class="switch">
                        <input type="checkbox" id="autorefresh-checkbox">
                        <span class="slider"></span>
                    </label>
                </div>
                
                <button class="btn" id="edit-html-btn" title="Edit Pasted HTML" style="display: none; background-color: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.4); color: #10b981;">
                    <i class="fa-solid fa-pen-to-square"></i> Edit HTML
                </button>
                
                <button class="btn" id="refresh-btn" title="Refresh File Content">
                    <i class="fa-solid fa-arrows-rotate"></i> Reload
                </button>
                
                <button class="btn" id="theme-btn" title="Toggle Dark/Light Mode">
                    <i class="fa-solid fa-sun"></i> Theme
                </button>
                
                <button class="btn btn-primary" id="print-btn" title="Export/Print as PDF">
                    <i class="fa-solid fa-file-pdf"></i> Print / PDF
                </button>
            </div>
        </header>

        <div class="viewer-container">
            <div class="document-card" id="document-card">
                <div class="empty-state" id="empty-state">
                    <i class="fa-solid fa-folder-open"></i>
                    <h2>No Document Loaded</h2>
                    <p>Select a markdown file from the sidebar to begin rendering.</p>
                </div>
                <div class="html-sandbox" id="html-sandbox" style="display: none;">
                    <h2 style="font-family: 'Outfit', sans-serif; font-weight: 600; margin-bottom: 5px;">Paste HTML Content</h2>
                    <p style="color: var(--text-secondary); margin-bottom: 15px; font-size: 0.9rem;">
                        Paste your raw HTML code (including MathML or LaTeX) below to render it with math parsing and image resolution.
                    </p>
                    <textarea id="html-input" placeholder="Type or paste HTML code here..."></textarea>
                    <div style="display: flex; gap: 10px;">
                        <button class="btn btn-primary" id="render-sandbox-btn" style="padding: 10px 20px; font-weight: 600; border-radius: 8px;">
                            <i class="fa-solid fa-play"></i> Render HTML
                        </button>
                        <button class="btn" id="clear-sandbox-btn" style="padding: 10px 20px; font-weight: 600; border-radius: 8px;">
                            <i class="fa-solid fa-trash-can"></i> Clear
                        </button>
                    </div>
                </div>
                <div class="markdown-body" id="rendered-content" style="display: none;">
                    <!-- Rendered markdown goes here -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let files = [];
        let currentFile = null;
        let autoRefreshInterval = null;

        function preprocessLaTeXFigures(text, isHTML = false) {
            if (!text) return text;
            let cleaned = text;
            if (isHTML) {
                cleaned = cleaned.replace(/\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/gi, '<img src="$1" alt="" />');
            } else {
                cleaned = cleaned.replace(/\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/gi, '![]($1)');
            }
            cleaned = cleaned.replace(/\\begin\{figure\}/gi, '');
            cleaned = cleaned.replace(/\\end\{figure\}/gi, '');
            cleaned = cleaned.replace(/\\begin\{center\}/gi, '');
            cleaned = cleaned.replace(/\\end\{center\}/gi, '');
            cleaned = cleaned.replace(/\\centering/gi, '');
            cleaned = cleaned.replace(/\\hfill/gi, '');
            cleaned = cleaned.replace(/\\hspace\*?\{[^}]*\}/gi, '');
            cleaned = cleaned.replace(/\\vspace\*?\{[^}]*\}/gi, '');
            cleaned = cleaned.replace(/\\label\{[^}]*\}/gi, '');
            cleaned = cleaned.replace(/\\captionsetup(?:\[[^\]]*\])?\{[^}]*\}/gi, '');
            if (isHTML) {
                cleaned = cleaned.replace(/\\caption(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/gi, '<p>$1</p>');
            } else {
                cleaned = cleaned.replace(/\\caption(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}/gi, '\n\n$1\n\n');
            }
            return cleaned;
        }

        // Dom Elements
        const fileListEl = document.getElementById('file-list');
        const searchInputEl = document.getElementById('search-input');
        const activeTitleEl = document.getElementById('active-file-title');
        const emptyStateEl = document.getElementById('empty-state');
        const contentEl = document.getElementById('rendered-content');
        const refreshBtn = document.getElementById('refresh-btn');
        const themeBtn = document.getElementById('theme-btn');
        const printBtn = document.getElementById('print-btn');
        const autoRefreshCheckbox = document.getElementById('autorefresh-checkbox');
        
        const pasteHtmlBtn = document.getElementById('paste-html-btn');
        const editHtmlBtn = document.getElementById('edit-html-btn');
        const htmlSandboxEl = document.getElementById('html-sandbox');
        const htmlInputEl = document.getElementById('html-input');
        const renderSandboxBtn = document.getElementById('render-sandbox-btn');
        const clearSandboxBtn = document.getElementById('clear-sandbox-btn');

        // Fetch file list on load
        async function fetchFileList() {
            try {
                const res = await fetch('/api/list');
                files = await res.json();
                renderFileList(files);
                
                // Auto-select first file if none selected
                // Auto-select first file if none selected
                if (files.length > 0 && !currentFile) {
                    // Try to auto-select full.md, combined.mmd, or screenshot first
                    const preferredFile = files.find(f => f.name === "full.md" || f.name === "combined.mmd" || f.name.includes("Screenshot"));
                    selectFile(preferredFile || files[0]);
                }
            } catch (err) {
                console.error("Error loading files list:", err);
            }
        }

        // Render file list with search filter
        function renderFileList(list) {
            fileListEl.innerHTML = '';
            if (list.length === 0) {
                fileListEl.innerHTML = '<div style="color: var(--text-secondary); text-align: center; margin-top: 20px; font-size: 0.9rem;">No .md or .mmd files found.</div>';
                return;
            }
            
            list.forEach(file => {
                const item = document.createElement('div');
                item.className = 'file-item';
                if (currentFile && currentFile.abs_path === file.abs_path) {
                    item.className += ' active';
                }
                
                item.innerHTML = `
                    <div class="file-name" title="${file.name}">${file.name}</div>
                    <div class="file-meta">
                        <span>Folder: ${file.dir}</span>
                    </div>
                `;
                
                item.addEventListener('click', () => selectFile(file));
                fileListEl.appendChild(item);
            });
        }

        // Search filtering
        searchInputEl.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = files.filter(f => f.name.toLowerCase().includes(query) || f.dir.toLowerCase().includes(query));
            renderFileList(filtered);
        });

        // Select file to render
        async function selectFile(file) {
            currentFile = file;
            
            // Remove active highlight from sandbox button
            pasteHtmlBtn.classList.remove('active');
            
            // Hide sandbox editor and edit button
            htmlSandboxEl.style.display = 'none';
            editHtmlBtn.style.display = 'none';
            
            // Highlight in list
            document.querySelectorAll('.file-item').forEach(item => {
                const name = item.querySelector('.file-name').innerText;
                if (name === file.name) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });

            activeTitleEl.innerHTML = `<i class="fa-solid fa-file-lines"></i> ${file.name}`;
            await loadMarkdown(file.abs_path);
        }

        // Load and render markdown from server
        async function loadMarkdown(path) {
            try {
                const res = await fetch(`/api/markdown?path=${encodeURIComponent(path)}`);
                if (!res.ok) throw new Error("Could not load file contents");
                const text = await res.text();
                
                emptyStateEl.style.display = 'none';
                contentEl.style.display = 'block';
                
                // Preprocess LaTeX figures in text
                const preprocessedText = preprocessLaTeXFigures(text, false);
                
                // Parse markdown using marked.js
                let parsedHTML = marked.parse(preprocessedText);
                
                // Set HTML content
                contentEl.innerHTML = parsedHTML;
                
                // Replace relative image URLs with the api-relative paths
                contentEl.querySelectorAll('img').forEach(img => {
                    const src = img.getAttribute('src');
                    if (src && !src.startsWith('http') && !src.startsWith('/') && !src.startsWith('data:')) {
                        const pathParam = currentFile ? currentFile.abs_path : '';
                        img.src = `/api/image?path=${encodeURIComponent(pathParam)}&img=${encodeURIComponent(src)}`;
                    }
                });

                // Render math equations using KaTeX
                renderMathInElement(contentEl, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\\\(', right: '\\\\)', display: false},
                        {left: '\\\\[', right: '\\\\]', display: true}
                    ],
                    throwOnError: false
                });

                // Highlight code blocks
                Prism.highlightAllUnder(contentEl);

            } catch (err) {
                console.error("Error loading markdown:", err);
                contentEl.innerHTML = `<div style="color: #ef4444; padding: 20px; font-weight: 500;"><i class="fa-solid fa-circle-exclamation"></i> Error loading file content: ${err.message}</div>`;
            }
        }

        // Refresh action
        refreshBtn.addEventListener('click', async () => {
            if (currentFile) {
                const originalBtn = refreshBtn.innerHTML;
                refreshBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate fa-spin"></i> Loading';
                await fetchFileList(); // Refresh the list of files first
                if (currentFile) {
                    await loadMarkdown(currentFile.abs_path);
                }
                setTimeout(() => {
                    refreshBtn.innerHTML = originalBtn;
                }, 400);
            }
        });

        // Print action
        printBtn.addEventListener('click', () => {
            window.print();
        });

        // Theme Toggle
        themeBtn.addEventListener('click', () => {
            const body = document.body;
            const isDark = body.classList.contains('dark-theme');
            const themeIcon = themeBtn.querySelector('i');
            
            if (isDark) {
                body.classList.remove('dark-theme');
                body.classList.add('light-theme');
                themeIcon.className = 'fa-solid fa-sun';
            } else {
                body.classList.remove('light-theme');
                body.classList.add('dark-theme');
                themeIcon.className = 'fa-solid fa-moon';
            }
        });

        // Auto Refresh setup
        autoRefreshCheckbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                autoRefreshInterval = setInterval(() => {
                    if (currentFile) {
                        loadMarkdown(currentFile.abs_path);
                    }
                }, 2000);
            } else {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }
        });

        // Paste HTML Sandbox trigger
        pasteHtmlBtn.addEventListener('click', () => {
            // Remove active from other files
            document.querySelectorAll('.file-item').forEach(item => item.classList.remove('active'));
            pasteHtmlBtn.classList.add('active');
            
            // Set header title
            activeTitleEl.innerHTML = `<i class="fa-solid fa-code"></i> Pasted HTML Sandbox`;
            
            // Hide elements and show sandbox textarea
            emptyStateEl.style.display = 'none';
            contentEl.style.display = 'none';
            htmlSandboxEl.style.display = 'flex';
            editHtmlBtn.style.display = 'none';
            
            // Stop auto-refresh since we are in sandbox
            if (autoRefreshCheckbox.checked) {
                autoRefreshCheckbox.checked = false;
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }
        });

        // Clear Sandbox
        clearSandboxBtn.addEventListener('click', () => {
            htmlInputEl.value = '';
        });

        // Render Sandbox HTML
        renderSandboxBtn.addEventListener('click', () => {
            const rawHTML = htmlInputEl.value;
            if (!rawHTML.trim()) {
                alert("Please enter some HTML code first!");
                return;
            }
            
            try {
                // Preprocess LaTeX figures in the raw HTML
                const preprocessedHTML = preprocessLaTeXFigures(rawHTML, true);
                
                // Set HTML content directly since it's already HTML
                contentEl.innerHTML = preprocessedHTML;
                
                // Replace relative image URLs with the api-relative paths
                contentEl.querySelectorAll('img').forEach(img => {
                    const src = img.getAttribute('src');
                    if (src && !src.startsWith('http') && !src.startsWith('/') && !src.startsWith('data:')) {
                        const pathParam = currentFile ? currentFile.abs_path : '';
                        img.src = `/api/image?path=${encodeURIComponent(pathParam)}&img=${encodeURIComponent(src)}`;
                    }
                });

                // Render math equations using KaTeX
                renderMathInElement(contentEl, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\\\(', right: '\\\\)', display: false},
                        {left: '\\\\[', right: '\\\\]', display: true}
                    ],
                    throwOnError: false
                });

                // Highlight code blocks
                Prism.highlightAllUnder(contentEl);

                // Show rendered content, hide sandbox editor
                htmlSandboxEl.style.display = 'none';
                contentEl.style.display = 'block';
                editHtmlBtn.style.display = 'inline-flex';
                
            } catch (err) {
                console.error("Error rendering sandbox HTML:", err);
                alert("Error rendering HTML: " + err.message);
            }
        });

        // Edit HTML back-trigger
        editHtmlBtn.addEventListener('click', () => {
            contentEl.style.display = 'none';
            htmlSandboxEl.style.display = 'flex';
            editHtmlBtn.style.display = 'none';
        });

        // Initial Load
        fetchFileList();
    </script>
</body>
</html>
"""

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

class MarkdownRenderHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Prevent logging spam in console
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            
        elif path == "/api/list":
            md_files = []
            if (os.path.exists(OUTPUT_DIR)):
                for root, dirs, files_in_dir in os.walk(OUTPUT_DIR):
                    for file in files_in_dir:
                        if file.endswith((".md", ".mmd")):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, OUTPUT_DIR)
                            
                            # Get directory parent path name
                            parent_dir_name = os.path.basename(os.path.dirname(full_path))
                            grandparent_dir_name = os.path.basename(os.path.dirname(os.path.dirname(full_path)))
                            display_dir = f"{grandparent_dir_name}/{parent_dir_name}" if grandparent_dir_name else parent_dir_name

                            md_files.append({
                                "name": file,
                                "rel_path": rel_path,
                                "abs_path": full_path,
                                "dir": display_dir
                            })
            
            # Sort files alphabetically
            md_files.sort(key=lambda x: x["name"])
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(md_files).encode("utf-8"))
            
        elif path == "/api/markdown":
            abs_path = query.get("path", [None])[0]
            if not abs_path:
                self.send_error(400, "Missing path parameter")
                return
            
            abs_path = os.path.abspath(abs_path)
            # Security check: Ensure file is under the BASE_DIR
            if not abs_path.startswith(BASE_DIR):
                self.send_error(403, "Access denied")
                return
                
            if not os.path.exists(abs_path):
                self.send_error(404, "File not found")
                return
                
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
            
        elif path == "/api/image":
            md_path = query.get("path", [None])[0]
            img_rel = query.get("img", [None])[0]
            
            if not img_rel:
                self.send_error(400, "Missing parameters")
                return
                
            img_rel = urllib.parse.unquote(img_rel)
            img_path = None
            
            if md_path and md_path.strip():
                md_path = os.path.abspath(md_path)
                if md_path.startswith(BASE_DIR):
                    md_dir = os.path.dirname(md_path)
                    candidate_path = os.path.abspath(os.path.join(md_dir, img_rel))
                    if candidate_path.startswith(BASE_DIR) and os.path.exists(candidate_path):
                        img_path = candidate_path

            # Fallback recursive search inside OUTPUT_DIR if not found yet
            if not img_path or not os.path.exists(img_path):
                img_filename = os.path.basename(img_rel)
                for root, dirs, files_in_dir in os.walk(OUTPUT_DIR):
                    if img_filename in files_in_dir:
                        candidate_path = os.path.join(root, img_filename)
                        if candidate_path.startswith(BASE_DIR):
                            img_path = candidate_path
                            break
                            
            if not img_path or not os.path.exists(img_path):
                self.send_error(404, "Image not found")
                return
                
            # Determine content type
            content_type = "image/jpeg"
            if img_path.lower().endswith(".png"):
                content_type = "image/png"
            elif img_path.lower().endswith(".svg"):
                content_type = "image/svg+xml"
            elif img_path.lower().endswith(".gif"):
                content_type = "image/gif"
            elif img_path.lower().endswith(".webp"):
                content_type = "image/webp"
                
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            
            with open(img_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "Not found")

def run(port=PORT):
    # Try starting on default port, otherwise pick a random free port
    server_address = ('127.0.0.1', port)
    try:
        httpd = HTTPServer(server_address, MarkdownRenderHandler)
        actual_port = port
    except OSError:
        actual_port = get_free_port()
        server_address = ('127.0.0.1', actual_port)
        httpd = HTTPServer(server_address, MarkdownRenderHandler)

    url = f"http://127.0.0.1:{actual_port}/"
    print(f"============================================================")
    print(f" Acadza Markdown Live Viewer is starting!")
    print(f" Server URL: {url}")
    print(f" Scanned Folder: {OUTPUT_DIR}")
    print(f"============================================================")
    
    # Auto-open browser window
    webbrowser.open(url)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    run()

document.addEventListener('DOMContentLoaded', function () {
    // Initialize terminal
    const terminal = new Terminal({
        cursorBlink: true,
        convertEol: true,  // Convert line feed characters to carriage return + line feed
        theme: {
            background: '#1e1e1e',
            foreground: '#f0f0f0',
            cursor: '#f0f0f0',
            black: '#000000',
            red: '#e06c75',
            green: '#98c379',
            yellow: '#d19a66',
            blue: '#61afef',
            magenta: '#c678dd',
            cyan: '#56b6c2',
            white: '#abb2bf',
            brightBlack: '#5c6370',
            brightRed: '#e06c75',
            brightGreen: '#98c379',
            brightYellow: '#d19a66',
            brightBlue: '#61afef',
            brightMagenta: '#c678dd',
            brightCyan: '#56b6c2',
            brightWhite: '#ffffff'
        },
        fontFamily: '"Ubuntu Mono", monospace',
        fontSize: 15,
        lineHeight: 1.2,
        rendererType: 'canvas',   // Use canvas renderer for better performance
        disableStdin: false
    });

    // Add fit addon to make terminal resize to container
    const fitAddon = new FitAddon.FitAddon();
    terminal.loadAddon(fitAddon);

    // Create socket connection
    const socket = io();
    // Store for global access
    window.term = terminal;
    window.fitAddon = fitAddon;

    // Open terminal
    terminal.open(document.getElementById('terminal'));
    fitAddon.fit();

    // Handle connection events
    socket.on('connect', () => {
        console.log('Connected to server');
        // Start polling for output updates when connected
        socket.emit('start_output_polling');
    });

    socket.on('disconnect', () => {
        terminal.write('\r\n\n[Connection lost. Reconnecting...]\r\n');
    });

    // Handle terminal output from server
    socket.on('terminal_output', (data) => {
        terminal.write(data.output);
    });

    // Send terminal input to server
    terminal.onData((data) => {
        socket.emit('terminal_input', {input: data});
    });

    // Handle window resize
    window.addEventListener('resize', () => {
        fitAddon.fit();
        // Notify server of new terminal size
        const dimensions = {
            cols: terminal.cols,
            rows: terminal.rows
        };
        socket.emit('resize', dimensions);
    });

    // Handle fullscreen toggle
    const fullscreenButton = document.getElementById('btn-fullscreen');
    const terminalElement = document.getElementById('terminal');

    fullscreenButton.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            terminalElement.requestFullscreen().catch(err => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    });

    // Fit terminal on fullscreen change
    document.addEventListener('fullscreenchange', () => {
        setTimeout(() => {
            fitAddon.fit();
            // Notify server of new terminal size
            const dimensions = {
                cols: terminal.cols,
                rows: terminal.rows
            };
            socket.emit('resize', dimensions);
        }, 100);
    });

    // Re-enable polling after page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            socket.emit('start_output_polling');
        } else {
            socket.emit('stop_output_polling');
        }
    });
});

// Function to fetch and display scan datasets
function loadScanDatasets() {
    fetch('/api/scans')
        .then(response => response.json())
        .then(data => {
            console.log('Got scans from API call:', data);
            return data; // Return the data to the next .then()
        })
        .then(data => {
            const scanContainer = document.getElementById('scan-datasets');
            scanContainer.innerHTML = '';

            if (data.length === 0) {
                scanContainer.innerHTML = '<p>No scan datasets available</p>';
                return;
            }

            data.forEach(scan => {
                const scanElement = document.createElement('div');
                scanElement.className = 'scan-item';
                scanElement.textContent = scan;
                scanElement.addEventListener('dblclick', () => {
                    // Handle scan selection by double click: write scan name in the terminal
                    const terminal = window.term;
                    if (terminal) {
                        terminal.write(`${scan}`);
                    }
                });
                scanContainer.appendChild(scanElement);
            });
        })
        .catch(error => {
            console.error('Error fetching scan datasets:', error);
            document.getElementById('scan-datasets').innerHTML =
                '<p>Error loading scan datasets</p>';
        });
}

// Load scan datasets when page loads
document.addEventListener('DOMContentLoaded', () => {
    // After terminal is initialized
    setTimeout(loadScanDatasets, 1000);

    // Add refresh button functionality
    document.getElementById('refresh-scans').addEventListener('click', loadScanDatasets);

    // Initialize TOML editor
    initTomlEditor();
});

document.addEventListener('DOMContentLoaded', function() {
    const resizeHandle = document.querySelector('.resize-handle');
    const terminalSide = document.querySelector('.terminal-side');
    const mainContainer = document.querySelector('.main-container');

    let isResizing = false;

    // Mouse down event on the resize handle
    resizeHandle.addEventListener('mousedown', function(e) {
        isResizing = true;
        resizeHandle.classList.add('active');

        // Prevent text selection during resize
        document.body.style.userSelect = 'none';

        // Initial mouse position
        const startX = e.clientX;
        const startWidth = terminalSide.offsetWidth;

        // Mouse move event for resizing
        function handleMouseMove(e) {
            if (!isResizing) return;

            const newWidth = startWidth + (e.clientX - startX);
            const containerWidth = mainContainer.offsetWidth;

            // Limit resizing within reasonable bounds (10% to 90% of container)
            const minWidth = Math.max(200, containerWidth * 0.1);
            const maxWidth = containerWidth * 0.9;

            if (newWidth >= minWidth && newWidth <= maxWidth) {
                terminalSide.style.width = newWidth + 'px';

                // Ensure terminal resizes properly if you're using xterm.js
                if (window.fitAddon) {
                    window.fitAddon.fit();
                }
            }
        }

        // Mouse up event to stop resizing
        function handleMouseUp() {
            isResizing = false;
            resizeHandle.classList.remove('active');
            document.body.style.userSelect = '';

            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        }

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    });
});

// CodeMirror instance
let codeMirror;

// Default TOML content
const defaultTomlContent = `# Example TOML configuration
[task]
name = 'Colmap'
type = 'reconstruction'

[parameters]
quality = 'high'
use_gpu = true
match_type = 'exhaustive'

[output]
format = 'ply'
save_intermediate = false`;

// TOML Editor Functionality
function initTomlEditor() {
    const editorContainer = document.getElementById('toml-editor');
    const filenameInput = document.getElementById('toml-filename');
    const saveButton = document.getElementById('save-toml-btn');
    const loadButton = document.getElementById('load-toml-btn');
    const dropZone = document.getElementById('toml-editor-container');
    const fileSelector = document.getElementById('toml-file-selector');
    const filesList = document.getElementById('toml-files-list');
    const closeFileSelector = document.getElementById('close-file-selector');

    let currentFilename = 'config.toml';

    // Initialize CodeMirror
    codeMirror = CodeMirror(editorContainer, {
        value: defaultTomlContent,
        mode: 'toml',
        theme: 'monokai',
        lineNumbers: true,
        indentUnit: 4,
        smartIndent: true,
        tabSize: 4,
        indentWithTabs: false,
        electricChars: true,
        lineWrapping: true,
        matchBrackets: true,
        autoCloseBrackets: true,
        autofocus: false
    });

    // Adjust editor size after initialization
    setTimeout(() => {
        codeMirror.refresh();
    }, 100);

    // Set initial filename
    filenameInput.value = currentFilename;

    // Handle file drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];

            // Check if it's a TOML file
            if (file.name.toLowerCase().endsWith('.toml')) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    codeMirror.setValue(event.target.result);
                    currentFilename = file.name;
                    filenameInput.value = currentFilename;
                };
                reader.readAsText(file);
            } else {
                alert('Please drop a TOML file (.toml)');
            }
        }
    });

    // Handle save button click
    saveButton.addEventListener('click', () => {
        // Get the updated filename from the input
        const filename = filenameInput.value.trim();
        if (!filename) {
            alert('Please enter a filename');
            return;
        }

        // Ensure filename has .toml extension
        const finalFilename = filename.toLowerCase().endsWith('.toml')
            ? filename
            : `${filename}.toml`;

        // Get the TOML content from CodeMirror
        const content = codeMirror.getValue();

        // Save the file
        saveTomlFile(finalFilename, content);
    });

    // Handle load button click
    loadButton.addEventListener('click', () => {
        // Show file selector and load the list of TOML files
        fileSelector.classList.remove('hidden');
        loadTomlFilesList();
    });

    // Close file selector
    closeFileSelector.addEventListener('click', () => {
        fileSelector.classList.add('hidden');
    });

    // Allow editing filename
    filenameInput.addEventListener('change', () => {
        currentFilename = filenameInput.value;
    });
}

// Function to load the list of TOML files
function loadTomlFilesList() {
    const filesList = document.getElementById('toml-files-list');
    filesList.innerHTML = '<div class="loading-indicator">Loading files...</div>';

    // Get the username from the DOM
    const usernameElement = document.querySelector('.username');
    let username = 'default';

    if (usernameElement) {
        // Extract username from format like "@username"
        const usernameText = usernameElement.textContent;
        username = usernameText.startsWith('@') ? usernameText.substring(1) : usernameText;
    }

    // Fetch the list of available TOML files
    fetch(`/api/list-toml-files?username=${encodeURIComponent(username.trim())}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load files list');
            }
            return response.json();
        })
        .then(data => {
            filesList.innerHTML = '';

            if (data.files && data.files.length > 0) {
                data.files.forEach(file => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    fileItem.innerHTML = `<i class="bi bi-file-earmark-text"></i> ${file}`;
                    fileItem.addEventListener('click', () => {
                        loadTomlFile(file);
                    });
                    filesList.appendChild(fileItem);
                });
            } else {
                filesList.innerHTML = '<div class="no-files">No TOML files found</div>';
            }
        })
        .catch(error => {
            console.error('Error loading TOML files list:', error);
            filesList.innerHTML = `<div class="no-files">Error: ${error.message}</div>`;
        });
}

// Function to load a specific TOML file
function loadTomlFile(filename) {
    // Get the username from the DOM
    const usernameElement = document.querySelector('.username');
    let username = 'default';

    if (usernameElement) {
        // Extract username from format like "@username"
        const usernameText = usernameElement.textContent;
        username = usernameText.startsWith('@') ? usernameText.substring(1) : usernameText;
    }

    fetch(`/api/load-toml-file?filename=${encodeURIComponent(filename)}&username=${encodeURIComponent(username.trim())}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to load file');
            }
            return response.json();
        })
        .then(data => {
            if (data.content) {
                // Set content in CodeMirror
                codeMirror.setValue(data.content);
                document.getElementById('toml-filename').value = filename;
                document.getElementById('toml-file-selector').classList.add('hidden');
            } else {
                throw new Error('File content is empty');
            }
        })
        .catch(error => {
            console.error('Error loading TOML file:', error);
            alert(`Error loading file: ${error.message}`);
        });
}

// Function to save TOML file to server
function saveTomlFile(filename, content) {
    // Get the username from the DOM
    const usernameElement = document.querySelector('.username');
    let username = 'default';

    if (usernameElement) {
        // Extract username from format like "@username"
        const usernameText = usernameElement.textContent;
        username = usernameText.startsWith('@') ? usernameText.substring(1) : usernameText;
    }

    fetch('/api/save-toml', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filename: filename,
            content: content,
            username: username.trim()
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to save file');
        }
        return response.json();
    })
    .then(data => {
        console.log('File saved successfully:', data);
        alert(`File "${filename}" saved successfully to ${data.path}`);
    })
    .catch(error => {
        console.error('Error saving file:', error);
        alert(`Error saving file: ${error.message}`);
    });
}

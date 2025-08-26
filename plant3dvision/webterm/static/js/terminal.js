document.addEventListener('DOMContentLoaded', function () {
    // Initialize terminal
    const terminal = new Terminal({
        cursorBlink: true,
        convertEol: false,
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
        rendererType: 'canvas',
        disableStdin: false,
        cursorStyle: 'block',
        cursorWidth: 1,
        // Enhanced settings for better terminal emulation
        allowTransparency: false,
        bellSound: undefined,
        bellStyle: 'none',
        fastScrollModifier: 'alt',
        scrollback: 1000,  // scrollback buffer
        windowsMode: false
    });

    // Add fit addon
    const fitAddon = new FitAddon.FitAddon();
    terminal.loadAddon(fitAddon);

    // Create socket connection with better error handling
    const socket = io({
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 5,
        timeout: 20000
    });

    // Terminal state management
    let terminalState = {
        connected: false,
        inputBuffer: '',
        commandHistory: [],
        historyIndex: -1,
        currentLine: '',
        cursorPosition: 0,
        isProcessingInput: false
    };

    // Store references globally
    window.term = terminal;
    window.fitAddon = fitAddon;
    window.socket = socket;

    // Open terminal
    terminal.open(document.getElementById('terminal'));
    fitAddon.fit();

    // Enhanced key event handler with support for all required features
    terminal.attachCustomKeyEventHandler((event) => {
        if (event.type !== 'keydown') {
            return true;
        }

        // Handle Ctrl+C specially
        if (event.ctrlKey && event.code === 'KeyC') {
            // Allow Ctrl+C to pass through to terminal
            return true;
        }

        // Handle Ctrl+V for paste
        if (event.ctrlKey && event.code === 'KeyV') {
            event.preventDefault();
            handlePaste();
            return false;
        }

        // Handle 'Ctrl+?' or 'Ctrl+,' for help (AZERTY keyboard support)
        if (event.ctrlKey && (event.code === 'Slash' || event.key === '?' || event.code === 'Comma' || event.key === ',')) {
            event.preventDefault();
            showKeyboardShortcuts();
            return false;
        }

        // Handle word navigation: Ctrl+Left/Right Arrow
        if (event.ctrlKey && (event.code === 'ArrowLeft' || event.code === 'ArrowRight')) {
            event.preventDefault();
            handleWordNavigation(event.code === 'ArrowRight');
            return false;
        }

        // Handle history navigation: Up/Down arrows
        if (event.code === 'ArrowUp' || event.code === 'ArrowDown') {
            event.preventDefault();
            handleHistoryNavigation(event.code === 'ArrowDown');
            return false;
        }

        // Handle left/right arrow for cursor movement
        if (event.code === 'ArrowLeft' || event.code === 'ArrowRight') {
            // Let these pass through to the shell
            return true;
        }

        // Handle Tab for completion
        if (event.code === 'Tab') {
            event.preventDefault();
            handleTabCompletion();
            return false;
        }

        // Handle Enter
        if (event.code === 'Enter') {
            // Let Enter pass through normally
            return true;
        }

        // Let other keys pass through
        return true;
    });

    // Handle middle-click paste
    terminal.element.addEventListener('mousedown', (event) => {
        if (event.button === 1) { // Middle mouse button
            event.preventDefault();
            handleMiddleClickPaste(event);
        }
    });

    // Handle regular paste functionality
    async function handlePaste() {
        try {
            const text = await navigator.clipboard.readText();
            if (text) {
                // Send pasted text to terminal
                socket.emit('terminal_input', { input: text });
            }
        } catch (err) {
            console.warn('Could not read clipboard:', err);
        }
    }

    // Handle middle-click paste (Linux-style)
    function handleMiddleClickPaste(event) {
        const selection = window.getSelection().toString();
        if (selection) {
            socket.emit('terminal_input', { input: selection });
        }
    }

    // Handle word navigation with Ctrl+Arrow keys
    function handleWordNavigation(forward) {
        // Send appropriate escape sequences for word navigation
        const sequence = forward ? '\x1b[1;5C' : '\x1b[1;5D'; // Ctrl+Right : Ctrl+Left
        socket.emit('terminal_input', { input: sequence });
    }

    // Handle command history navigation
    function handleHistoryNavigation(down) {
        // Send appropriate escape sequences for history navigation
        const sequence = down ? '\x1b[B' : '\x1b[A'; // Down : Up arrow
        socket.emit('terminal_input', { input: sequence });
    }

    // Handle tab completion
    function handleTabCompletion() {
        // Send tab character to trigger shell completion
        socket.emit('terminal_input', { input: '\t' });
    }

    // Enhanced connection handling
    socket.on('connect', () => {
        console.log('Connected to server');
        terminalState.connected = true;
        terminal.write('\r\n\x1b[32m[Connected to server]\x1b[0m\r\n');
        socket.emit('start_output_polling');
    });

    socket.on('disconnect', (reason) => {
        console.log('Disconnected from server:', reason);
        terminalState.connected = false;
        terminal.write('\r\n\x1b[31m[Connection lost. Reason: ' + reason + ']\x1b[0m\r\n');

        if (reason === 'io server disconnect') {
            // Server disconnected us, try to reconnect
            socket.connect();
        }
    });

    socket.on('reconnect', (attemptNumber) => {
        console.log('Reconnected after', attemptNumber, 'attempts');
        terminalState.connected = true;
        terminal.write('\r\n\x1b[32m[Reconnected to server]\x1b[0m\r\n');
        socket.emit('start_output_polling');
    });

    socket.on('reconnect_error', (error) => {
        console.error('Reconnection failed:', error);
    });

    // Handle terminal output with better error handling
    socket.on('terminal_output', (data) => {
        if (data && data.output) {
            try {
                terminal.write(data.output);
            } catch (error) {
                console.error('Error writing to terminal:', error);
            }
        }
    });

    // Enhanced input handling with buffering
    terminal.onData((data) => {
        if (!terminalState.connected) {
            terminal.write('\r\n\x1b[31m[Not connected to server]\x1b[0m\r\n');
            return;
        }

        // Add input debouncing for rapid typing
        if (terminalState.isProcessingInput) {
            terminalState.inputBuffer += data;
            return;
        }

        terminalState.isProcessingInput = true;

        const inputToSend = terminalState.inputBuffer + data;
        terminalState.inputBuffer = '';

        socket.emit('terminal_input', { input: inputToSend });

        // Reset processing flag after a short delay
        setTimeout(() => {
            terminalState.isProcessingInput = false;

            // Process any buffered input
            if (terminalState.inputBuffer) {
                const bufferedInput = terminalState.inputBuffer;
                terminalState.inputBuffer = '';
                socket.emit('terminal_input', { input: bufferedInput });
            }
        }, 10);
    });

    // Enhanced resize handling with debouncing
    let resizeTimeout;
    function handleResize() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            try {
                fitAddon.fit();
                const dimensions = {
                    cols: terminal.cols,
                    rows: terminal.rows
                };
                socket.emit('resize', dimensions);
                console.log('Terminal resized to:', dimensions);
            } catch (error) {
                console.error('Error during resize:', error);
            }
        }, 100);
    }

    // Window resize event
    window.addEventListener('resize', handleResize);

    // Handle fullscreen functionality
    const fullscreenButton = document.getElementById('btn-fullscreen');
    const terminalElement = document.getElementById('terminal');

    if (fullscreenButton) {
        fullscreenButton.addEventListener('click', () => {
            if (!document.fullscreenElement) {
                terminalElement.requestFullscreen().then(() => {
                    setTimeout(handleResize, 100);
                }).catch(err => {
                    console.error(`Error attempting to enable fullscreen: ${err.message}`);
                });
            } else {
                document.exitFullscreen();
            }
        });
    }

    // Handle fullscreen changes
    document.addEventListener('fullscreenchange', () => {
        setTimeout(handleResize, 100);
    });

    // Enhanced visibility change handling
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            if (terminalState.connected) {
                socket.emit('start_output_polling');
            }
        } else {
            socket.emit('stop_output_polling');
        }
    });

    // Help tooltip functionality
    const helpButton = document.getElementById('btn-help');
    const helpTooltip = document.getElementById('help-tooltip');
    const closeHelpTooltip = document.getElementById('close-help-tooltip');

    // Show help tooltip when hovering over help button
    if (helpButton && helpTooltip) {
        helpButton.addEventListener('mouseenter', () => {
            helpTooltip.classList.remove('hidden');
        });

        // Keep tooltip visible when hovering over it
        helpTooltip.addEventListener('mouseenter', () => {
            helpTooltip.classList.remove('hidden');
        });

        // Hide tooltip when mouse leaves both button and tooltip
        helpButton.addEventListener('mouseleave', (e) => {
            setTimeout(() => {
                if (!helpTooltip.matches(':hover') && !helpButton.matches(':hover')) {
                    helpTooltip.classList.add('hidden');
                }
            }, 100);
        });

        helpTooltip.addEventListener('mouseleave', () => {
            if (!helpButton.matches(':hover')) {
                helpTooltip.classList.add('hidden');
            }
        });

        // Also show on click
        helpButton.addEventListener('click', (e) => {
            e.preventDefault();
            helpTooltip.classList.toggle('hidden');
        });

        // Close button functionality
        closeHelpTooltip.addEventListener('click', () => {
            helpTooltip.classList.add('hidden');
        });
    }

    // Enhanced keyboard shortcut help - both Ctrl+? and terminal display
    function showKeyboardShortcuts() {
        const shortcuts = [
            'Ctrl+C: Interrupt current command',
            'Ctrl+V: Paste from clipboard',
            'Ctrl+? or Ctrl+,: Show this help',
            'Tab: Auto-complete commands/paths',
            'Up/Down: Navigate command history',
            'Ctrl+Left/Right: Move cursor by word',
            'Middle-click: Paste selection'
        ];

        terminal.write('\r\n\x1b[36m=== Keyboard Shortcuts ===\x1b[0m\r\n');
        shortcuts.forEach(shortcut => {
            terminal.write(`\x1b[33m${shortcut}\x1b[0m\r\n`);
        });
        terminal.write('\x1b[36m==========================\x1b[0m\r\n');
    }

    // Global keyboard shortcut handler
    // Only handle specific cases that don't overlap with terminal events
    window.addEventListener('keydown', (event) => {
        // Handle help shortcut ONLY when terminal is not focused
        if (event.ctrlKey && (event.code === 'Slash' || event.key === '?' || event.code === 'Comma' || event.key === ',')) {
            // Check if terminal is focused
            const terminalFocused = document.activeElement === terminal.textarea ||
                                   terminal.element.contains(document.activeElement) ||
                                   document.activeElement === terminal.element;

            if (!terminalFocused) {
                event.preventDefault();
                helpTooltip.classList.toggle('hidden');
            }
            // If terminal is focused, let the terminal's attachCustomKeyEventHandler handle it
        }

        // Close help tooltip with Escape key
        if (event.code === 'Escape' && !helpTooltip.classList.contains('hidden')) {
            event.preventDefault();
            helpTooltip.classList.add('hidden');
        }
    });

    // Close help tooltip when clicking outside
    document.addEventListener('click', (e) => {
        if (!helpTooltip.contains(e.target) && !helpButton.contains(e.target)) {
            helpTooltip.classList.add('hidden');
        }
    });

    // Cleanup function for page unload
    window.addEventListener('beforeunload', () => {
        socket.emit('stop_output_polling');
        socket.disconnect();
    });

    // Error handling for socket errors
    socket.on('error', (error) => {
        console.error('Socket error:', error);
        terminal.write('\r\n\x1b[31m[Socket error: ' + error.message + ']\x1b[0m\r\n');
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

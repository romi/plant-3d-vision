document.addEventListener('DOMContentLoaded', function() {
    // Initialize terminal
    const terminal = new Terminal({
        cursorBlink: true,
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
        fontFamily: 'Menlo, Monaco, "Courier New", monospace',
        fontSize: 14,
        lineHeight: 1.2
    });

    // Add fit addon to make terminal resize to container
    const fitAddon = new FitAddon.FitAddon();
    terminal.loadAddon(fitAddon);

    // Create socket connection
    const socket = io();

    // Open terminal
    terminal.open(document.getElementById('terminal'));
    fitAddon.fit();

    // Handle connection events
    socket.on('connect', () => {
        console.log('Connected to server');
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
        socket.emit('terminal_input', { input: data });
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
});
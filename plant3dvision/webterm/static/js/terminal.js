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
                scanElement.addEventListener('click', () => {
                    // Handle scan selection - could execute a command in terminal
                    const terminal = window.term;
                    if (terminal) {
                        terminal.write(`\r\nSelected scan: ${scan}\r\n`);
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
});
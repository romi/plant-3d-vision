document.addEventListener('DOMContentLoaded', function() {
    // Terminal Setup
    const term = new Terminal({
        cursorBlink: true,
        cursorStyle: 'block',
        fontSize: 14,
        fontFamily: 'Menlo, Monaco, "Courier New", monospace',
        theme: {
            background: '#282c34',
            foreground: '#abb2bf',
            cursor: '#abb2bf',
            cursorAccent: '#282c34',
            black: '#282c34',
            brightBlack: '#5c6370',
            red: '#e06c75',
            brightRed: '#be5046',
            green: '#98c379',
            brightGreen: '#7a9f60',
            yellow: '#e5c07b',
            brightYellow: '#d19a66',
            blue: '#61afef',
            brightBlue: '#3b84c0',
            magenta: '#c678dd',
            brightMagenta: '#9a52af',
            cyan: '#56b6c2',
            brightCyan: '#3c909b',
            white: '#abb2bf',
            brightWhite: '#828997'
        },
        allowTransparency: true,
        scrollback: 1000,
        letterSpacing: 0,
        lineHeight: 1.2,
        bellStyle: 'sound',
        copyOnSelect: true,
        rightClickSelectsWord: true
    });

    // Add necessary addons
    const fitAddon = new FitAddon.FitAddon();
    const webLinksAddon = new WebLinksAddon.WebLinksAddon();
    const socket = io();

    term.loadAddon(fitAddon);
    term.loadAddon(webLinksAddon);

    // Handle terminal resizing
    function handleResize() {
        fitAddon.fit();
        const dims = term.rows && term.cols ? {
            rows: term.rows,
            cols: term.cols
        } : fitAddon.proposeDimensions();

        if (dims) {
            socket.emit('terminal_resize', {
                rows: dims.rows,
                cols: dims.cols
            });
        }
    }

    // Initialize terminal
    term.open(document.getElementById('terminal'));
    handleResize();

    // Add window resize listener with debouncing
    let resizeTimeout;
    window.addEventListener('resize', () => {
        if (resizeTimeout) {
            clearTimeout(resizeTimeout);
        }
        resizeTimeout = setTimeout(handleResize, 100);
    });

    // Add clipboard handling
    term.attachCustomKeyEventHandler((event) => {
        // Handle Ctrl+C or Command+C (Mac) for copy
        if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
            const selection = term.getSelection();
            if (selection) {
                navigator.clipboard.writeText(selection);
                return false;
            }
        }
        // Handle Ctrl+V or Command+V (Mac) for paste
        if ((event.ctrlKey || event.metaKey) && event.key === 'v') {
            navigator.clipboard.readText().then(text => {
                if (text) {
                    socket.emit('terminal_input', { input: text });
                }
            });
            return false;
        }
        return true;
    });

    // Direct terminal input handling
    term.onData(data => {
        socket.emit('terminal_input', {
            input: data
        });
    console.log('Sent terminal input:', data);
    });

    // Terminal output handling
    socket.on('terminal_output', (data) => {
        console.log('Received terminal output:', data);
        if (data.error) {
            term.write(`\r\n\x1b[1;31mError:\x1b[0m ${data.error}\r\n`);
        } else {
            term.write(data.output);
        }
    });

    // Mouse selection handler
    term.element.addEventListener('mouseup', () => {
        const selection = term.getSelection();
        if (selection) {
            navigator.clipboard.writeText(selection);
        }
    });
});
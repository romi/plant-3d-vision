/**
 * Retrieves the API prefix from the global `window.WEBTERM_PREFIX` variable.
 *
 * If the prefix is not set or empty, an empty string is returned. The function
 * ensures that the prefix starts with a leading '/' and does not end with a trailing '/'.
 *
 * @return {string} The formatted app API prefix.
 */
function getApiPrefix() {
    // Get URL prefix from global variable or default to empty string
    let urlPrefix = window.WEBTERM_PREFIX || '';
    if (!urlPrefix) {
        return ''
    }
    // Add leading '/' if missing, remove trailing '/' if existing
    if (!urlPrefix.startsWith('/')) {
        urlPrefix = '/' + urlPrefix;
    }
    if (urlPrefix.endsWith('/')) {
        urlPrefix = urlPrefix.slice(0, -1);
    }
    //console.log('Using URL prefix:', urlPrefix);
    return urlPrefix;  // Return the formatted API prefix
}


/**
 * Constructs and returns the full URL for the app API endpoint.
 *
 * @return {string} The complete API URL including the prefix and '/api' suffix.
 */
function getApiUrl() {
    // Get the formatted API prefix
    let urlPrefix = getApiPrefix();
    // Append '/api' to the URL prefix to form the full API URL
    urlPrefix = `${urlPrefix}/api`;
    //console.log('Using API URL:', urlPrefix);
    return urlPrefix;  // Return the complete API URL
}

/**
 * Retrieves the URL for accessing the PlantDB REST API.
 *
 * @return {string} The constructed URL for the PlantDB REST API.
 */
function getPlantdbUrl() {
    // Try to get PLANTDB_API (URL pointing to a PlantDB REST API) or use local api (see '/api/scans' route)
    let baseUrl = window.PLANTDB_API || getApiUrl();

    // If baseUrl is not set, default to the local API URL
    if (!baseUrl) {
        baseUrl = getApiUrl();
        //console.log("Using local API:", baseUrl);
    } else {
        //console.log('Using PlantDB REST API URL:', baseUrl);
        // Clean up URL by removing trailing slash and consecutive slashes (except for those after protocol)
        baseUrl = `${baseUrl.replace(/\/$/, '')}`.replace(/([^:]\/)\/+/g, '$1');
    }

    return baseUrl;  // Return the cleaned PlantDB API URL
}


/**
 * Toggles the visibility of a password input field.
 *
 * @param {string} inputId - The ID of the password input element to be toggled.
 * @return {void}
 */
function togglePasswordVisibility(inputId) {
    const passwordInput = document.getElementById(inputId);
    const toggleIcon = passwordInput.nextElementSibling.querySelector('i');

    if (passwordInput.type === "password") {
        passwordInput.type = "text";
        toggleIcon.classList.remove('bi-eye');
        toggleIcon.classList.add('bi-eye-slash');
    } else {
        passwordInput.type = "password";
        toggleIcon.classList.remove('bi-eye-slash');
        toggleIcon.classList.add('bi-eye');
    }
}

/**
 * Retrieves the API prefix from the global `window.URL_PREFIX` variable.
 *
 * If the prefix is not set or empty, an empty string is returned. The function
 * ensures that the prefix starts with a leading '/' and does not end with a trailing '/'.
 *
 * @return {string} The formatted app API prefix.
 */
function getApiPrefix() {
    let urlPrefix = window.URL_PREFIX || '';
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
    console.log('Using URL prefix:', urlPrefix);
    return urlPrefix;
}


/**
 * Constructs and returns the full URL for the app API endpoint.
 *
 * @return {string} The complete API URL including the prefix and '/api' suffix.
 */
function getApiUrl() {
    let urlPrefix = getApiPrefix()
    urlPrefix = `${urlPrefix}/api`
    console.log('Using API URL:', urlPrefix);
    return urlPrefix;
}

/**
 * Retrieves the URL for accessing the PlantDB REST API.
 *
 * @return {string} The constructed URL for the PlantDB REST API.
 */
function getPlantdbUrl() {
    // Try to get the
    let baseUrl = window.PLANTDB_API || '';
    if (!baseUrl) {
        baseUrl = getApiUrl()
    } else {
        let urlPrefix = getApiPrefix()
        // Ensure no double slashes in URL
        baseUrl = `${baseUrl.replace(/\/$/, '')}${urlPrefix}`.replace(/([^:]\/)\/+/g, '$1');
    }
    console.log('Using PlantDB REST API URL:', baseUrl);
    return baseUrl;
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

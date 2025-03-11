Dropzone.autoDiscover = false;

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = new Dropzone("#upload-form", {
        url: "/upload",
        maxFiles: 1,
        acceptedFiles: ".zip",
        createImageThumbnails: false,
        init: function() {
            this.on("error", function(file, errorMessage) {
                console.error(errorMessage);
                alert(errorMessage.error || errorMessage);
            });

            this.on("success", function(file, response) {
                console.log("Upload successful:", response);
                alert(`File uploaded successfully. Scan ID: ${response.scan_id}`);
            });
        }
    });
});
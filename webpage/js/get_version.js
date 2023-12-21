// This script connects to the pypi.org server and checks the latest available version of Biogeme

const CACHE_DURATION = 24 * 60 * 60 * 1000; // 1 day in milliseconds

function fetchVersion() {
    fetch('https://pypi.org/pypi/biogeme/json')
    .then(response => response.json())
    .then(data => {
        const version = data.info.version;
        document.getElementById('version').textContent = version;

        // Cache the version and current timestamp
        localStorage.setItem('biogemeVersion', version);
        localStorage.setItem('biogemeTimestamp', Date.now());
    })
    .catch(error => {
        console.error('Error fetching the version:', error);
        document.getElementById('version').textContent = 'Error fetching version';
    });
}

// Check if we have a cached version and it's still valid
const cachedVersion = localStorage.getItem('biogemeVersion');
const cachedTimestamp = localStorage.getItem('biogemeTimestamp');
if (cachedVersion && cachedTimestamp && Date.now() - cachedTimestamp < CACHE_DURATION) {
    // Use the cached version
    document.getElementById('version').textContent = cachedVersion;
} else {
    // Fetch the latest version
    fetchVersion();
}

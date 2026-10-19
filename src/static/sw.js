const CACHE_NAME = 'artverificator-v1';
const urlsToCache = [
  '/',
  '/static/css/index.css',
  '/static/css/result.css',
  '/static/css/sobre.css',
  '/static/css/dataset.css',
  'static/icons/appstore-images/android/launchericon-192x192.png',
  'static/icons/appstore-images/android/launchericon-512x512.png',
];

// Instalação do Service Worker
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Cache aberto');
        return cache.addAll(urlsToCache);
      })
  );
});

// Intercepta requisições
self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Retorna do cache ou faz requisição normal
        return response || fetch(event.request);
      })
  );
});

// Atualização do Service Worker
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
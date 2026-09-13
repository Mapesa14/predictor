# The mobile app

One codebase. The React app in `web/` is the web app, the installable PWA and
the Android app — Capacitor wraps the same build in a native shell rather than
duplicating it in Kotlin or React Native.

## What exists

| | state |
|---|---|
| PWA (installable from a browser) | ready — manifest, icons, service worker |
| Android project (`web/android/`) | scaffolded, icons and splash generated |
| Android APK | **needs the Android SDK — see below** |
| iOS project | not added; requires macOS and an Apple Developer account |

## Build an APK

Everything but the last step is done. You need the Android SDK, which this
machine does not have:

1. Install **Android Studio** (it brings the SDK, platform-tools and an
   emulator), or the standalone `commandline-tools` plus `platform-tools` and
   `platforms;android-36`.
2. Point the project at it — either set `ANDROID_HOME`, or create
   `web/android/local.properties` with one line:
   `sdk.dir=C\:\\Users\\<you>\\AppData\\Local\\Android\\Sdk`
3. Then:

```bash
cd web
VITE_API_BASE=https://your-api-host npm run mobile:sync
cd android && ./gradlew assembleDebug
```

The APK lands in `web/android/app/build/outputs/apk/debug/`. `npm run
mobile:open` opens the project in Android Studio instead, which is easier for
running on a device and for producing a signed release build.

A note on the JDK: this machine has Java 25. The Android Gradle Plugin
generally tracks an older LTS, so if Gradle complains about the Java version,
install JDK 21 and point `org.gradle.java.home` at it in
`web/android/gradle.properties`.

## The one thing that will break it

**The app and the service are separate.** The phone holds the UI; the model
runs on a server. Inside the WebView the bundle is served from
`https://localhost`, so a relative `/api/...` resolves to the handset and hits
nothing. Two rules follow:

- `VITE_API_BASE` must be set at build time (or the address entered in the
  app's **Settings** screen, which overrides it at runtime and is how you point
  a test build at a laptop on the LAN).
- It must be **https**. Android blocks cleartext HTTP by default, and the
  request fails silently — the symptom is an app where every screen is simply
  empty, which looks like a broken model rather than a missing certificate.

A native build with no address configured opens on Settings rather than on an
empty fixture list, so this failure announces itself.

## What the native shell adds

- **Hardware back** goes up one screen and only exits from the root. Without
  handling it, Android's back button closes the app from anywhere.
- **Safe-area padding** (`html.native` in `index.css`) keeps the masthead out
  from under the status bar and the footer off the gesture bar.
- **No service worker.** Capacitor already serves the bundle from the device;
  a second cache layer only gives a stale build a way to survive an app update.
- **Larger touch targets** on the tab strip and buttons.

## Icons

`python web/tools/icons.py` regenerates everything from one drawn source:
192/512 PNGs and maskable variants for the manifest, an apple-touch-icon, and
the 1024 sources `npx capacitor-assets generate --android` expands into every
Android density. Maskable icons matter — launchers crop to whatever shape the
phone uses, so the artwork sits inside the centre 66%.

## Store policy, before you publish

Google Play allows informational prediction apps but prohibits navigational
elements that are a "call to action" to wager — buttons, tabs or webviews into
a bookmaker. Keeping affiliate links out is therefore a product decision, not
just a taste one. Both stores also take 15–30% on in-app digital
subscriptions; selling through mobile money outside the app avoids that but has
its own policy nuance, so settle it before setting a price.

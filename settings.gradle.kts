pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "ug-kotlin"

listOf(
    "../half-kotlin",
).forEach { path ->
    if (file(path).exists()) {
        includeBuild(path)
    }
}

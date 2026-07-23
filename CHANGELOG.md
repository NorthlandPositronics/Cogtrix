# Changelog

## [0.1.1](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.0...v0.1.1) (2026-02-16)


### Features

* activity indicator (spinner) during LLM processing ([0d9b4ac](https://github.com/NorthlandPositronics/Cogtrix/commit/0d9b4ac3a2ecf944bf65bf2bf40fde1a221980a9))
* auto version bump and release on merge to main ([f420524](https://github.com/NorthlandPositronics/Cogtrix/commit/f420524b6ada84a26de3abb1ab3d854c5ad21c0e))
* double spinner phrases, gradient color, trailing space ([7cc2067](https://github.com/NorthlandPositronics/Cogtrix/commit/7cc2067f8317e0743bb2227d47d40721bf806dcd))
* expand spinner to 80 phrases with humorous tech messages ([fd9d569](https://github.com/NorthlandPositronics/Cogtrix/commit/fd9d5699b84c9a1f13c481dd4d22c64450c0c3cb))
* hybrid /think pipeline with category-aware prompts ([6308a86](https://github.com/NorthlandPositronics/Cogtrix/commit/6308a86f28e58c8d225a79fb7b461c17b5b40fe5))
* publish multi-arch Docker image to GHCR ([d4325e4](https://github.com/NorthlandPositronics/Cogtrix/commit/d4325e41254a128e4588fbc173ce6cdee638e611))
* publish multi-arch Docker image to GHCR on main push ([1a28ea6](https://github.com/NorthlandPositronics/Cogtrix/commit/1a28ea60166a037305c5ba19bb1478bbaac3de0a))
* randomize spinner phrase order on each start ([0e0b107](https://github.com/NorthlandPositronics/Cogtrix/commit/0e0b10778577dc59a367c4a21c8fe522ccb5de69))
* replace custom version-bump with release-please ([5790406](https://github.com/NorthlandPositronics/Cogtrix/commit/57904061c3816c6770de27847c37585616b1ea0e))
* sequential intro phrases then random fun phrases ([1f5bbf9](https://github.com/NorthlandPositronics/Cogtrix/commit/1f5bbf9d207be12872cf719216b245a03cfe9b88))
* show spinner during forced deep_think invocations ([6d6679e](https://github.com/NorthlandPositronics/Cogtrix/commit/6d6679ebd0bd503839a81c239dab8a94b6f2803f))


### Bug Fixes

* add nosec markers for Bandit false positives (B311, B110) ([739449b](https://github.com/NorthlandPositronics/Cogtrix/commit/739449b6b63af7feb87dc708790792716b6805aa))
* category-aware Stage 2 framing for /think pipeline ([875f085](https://github.com/NorthlandPositronics/Cogtrix/commit/875f08541712645131531f38c699716e8d83e7ca))
* CI pipeline — dev dependencies, OSV-scanner, pyright error ([3e21dd1](https://github.com/NorthlandPositronics/Cogtrix/commit/3e21dd16079c971d9dcb97334cefa32db126d8b1))
* CI pipeline — dev dependencies, OSV-scanner, pyright error ([93ac20a](https://github.com/NorthlandPositronics/Cogtrix/commit/93ac20aa415dcfcfd22c1690370a6085d8c78b7a))
* classifier label normalization and tool output error filter ([f87821f](https://github.com/NorthlandPositronics/Cogtrix/commit/f87821fe0fc8e2779684506f3caafc7317d2f4dd))
* detect unconfigured provider before LLM init ([8425b93](https://github.com/NorthlandPositronics/Cogtrix/commit/8425b9331a72ba46262c7ba560fbdf25038a3027))
* graceful error messages for common configuration issues ([a9922e3](https://github.com/NorthlandPositronics/Cogtrix/commit/a9922e30e08042bcf967686207aaf9943e048e11))
* graceful error messages for common configuration issues ([00e8128](https://github.com/NorthlandPositronics/Cogtrix/commit/00e81281ac993d366a2231f8f81f243bad0c6e5c))
* pass force as boolean in version-bump ref update ([d11d4dd](https://github.com/NorthlandPositronics/Cogtrix/commit/d11d4dd70051f9a2589a91aa61bda0bc69a896e4))
* pause spinner during deep_think progress output ([a8dd126](https://github.com/NorthlandPositronics/Cogtrix/commit/a8dd126d29843065ac127e067b2306d0d6f498be))
* prevent deep_think from producing meta-descriptions instead of answers ([a3fd699](https://github.com/NorthlandPositronics/Cogtrix/commit/a3fd699104d9584163b4f6bcf8eb0349dc9c85be))
* remove contradictory 'do not invent facts' from synthesis categories ([1f2e65e](https://github.com/NorthlandPositronics/Cogtrix/commit/1f2e65eecfafed94cabb24a1ea0b42cc86ba2de6))
* remove skip condition and add actions permission ([4caffd0](https://github.com/NorthlandPositronics/Cogtrix/commit/4caffd02e10ab3c4a6eefd68ed1cad89d05ec6a2))
* rename release-please config to expected filename ([0bb0a0c](https://github.com/NorthlandPositronics/Cogtrix/commit/0bb0a0cc61ea8f964841647153a5120d707cae86))
* show spinner during /think slash command ([bbb5b56](https://github.com/NorthlandPositronics/Cogtrix/commit/bbb5b564f55fb65864fa7cd3c83805c1eb8a10f1))
* spinner line not clearing between frames ([2236607](https://github.com/NorthlandPositronics/Cogtrix/commit/2236607f35450dd05494e941b392d905f38675ef))
* suppress primp native stderr leaking into spinner output ([2fa5531](https://github.com/NorthlandPositronics/Cogtrix/commit/2fa55310181a2261d373323c0417ac1c266f5a2d))
* suppress ruff F401 for TYPE_CHECKING-only import in safety.py ([d9ec409](https://github.com/NorthlandPositronics/Cogtrix/commit/d9ec409294dbac181673129e7ca9c448e1d1a0df))
* use native ARM64 Linux runner for Docker build ([d95b140](https://github.com/NorthlandPositronics/Cogtrix/commit/d95b1408da9f96bfedcc741fc49c1827540b1393))
* use temp files for GitHub API blob creation in version-bump ([a2bc546](https://github.com/NorthlandPositronics/Cogtrix/commit/a2bc5464e9fd3afff44352dbcf05018ff0b8e6d8))

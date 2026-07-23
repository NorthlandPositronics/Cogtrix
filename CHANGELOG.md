# Changelog

> **Note (2026-05-24): Git history rewrite.** On 2026-05-24 the `main` and `next` branches were replaced with a single squashed snapshot of the then-current tree. The rewrite was needed to remove internal documentation — Architecture Decision Records, bug-investigation notes, internal roadmaps, security-audit material, team-process docs, and a private API test specification — from public git
> history. Those files had been moved out of the public working tree in PRs #1757 / #1758 and now live in the private `NorthlandPositronics/cogtrix-docs` submodule at `docs/optional/`, but they remained reachable via `git log -p` / `git show` against any commit prior to the cleanup. The squash to an orphan root commit makes them unreachable from any public ref.
>
> **Impact on this changelog:** entries below describe the features and fixes that actually shipped and remain accurate as a release record. The commit SHAs they reference, however, no longer resolve on `origin/main` or `origin/next` — those commits still exist as unreachable objects until git's garbage collection runs but are no longer in any branch's history.
>
> **Action for contributors with a local clone:**
>
> ```bash
> git fetch origin --prune
> git checkout main && git reset --hard origin/main
> git checkout next && git reset --hard origin/next
> git submodule update --init --recursive
> ```
>
> Any in-flight feature branch will need to be rebased onto (or cherry-picked into) the new orphan root before it can be merged.

## [0.5.0](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.4.1...v0.5.0) (2026-06-26)

### Features

- cut v0.5.0 release ([#2286](https://github.com/NorthlandPositronics/Cogtrix/issues/2286)) ([0f038b1](https://github.com/NorthlandPositronics/Cogtrix/commit/0f038b1afb4075eccc8c161555ddceed3c140ff0))

## [0.4.1](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.4.0...v0.4.1) (2026-06-21)

### Bug Fixes

- v0.4.1 release ([#2179](https://github.com/NorthlandPositronics/Cogtrix/issues/2179)) ([f29c4be](https://github.com/NorthlandPositronics/Cogtrix/commit/f29c4bed11804af47dcff609908427e83266ca42))

## [0.4.0](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.3.0...v0.4.0) (2026-06-19)

### Features

- v0.4.0 release ([#2109](https://github.com/NorthlandPositronics/Cogtrix/issues/2109)) ([f5e758a](https://github.com/NorthlandPositronics/Cogtrix/commit/f5e758ab8865687067f6e595eb6d7ff36ba1b6d5))

## [0.3.0](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.13...v0.3.0) (2026-06-06)

### Features

- v0.3.0 release ([#2036](https://github.com/NorthlandPositronics/Cogtrix/issues/2036)) ([ed033f8](https://github.com/NorthlandPositronics/Cogtrix/commit/ed033f8c79c8ffae292999bdb5364b6e53dcd3f9))

## [0.2.13](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.12...v0.2.13) (2026-05-31)

### Features

- v0.3.0 release — resolver/dispatcher hardening, ThreadPoolExecutor migration, agent fleet runner, rate-limit productionization ([#1936](https://github.com/NorthlandPositronics/Cogtrix/issues/1936)) ([98b4a81](https://github.com/NorthlandPositronics/Cogtrix/commit/98b4a814f7a958653dcd31c55715e4030e95ca4a))

## [0.2.12](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.11...v0.2.12) (2026-05-27)

### Bug Fixes

- cut v0.2.12 release + production release-title guard ([#1852](https://github.com/NorthlandPositronics/Cogtrix/issues/1852)) ([#1858](https://github.com/NorthlandPositronics/Cogtrix/issues/1858)) ([12ab8a7](https://github.com/NorthlandPositronics/Cogtrix/commit/12ab8a72c85c826e077ab96b116d33e26dff300b))

## [0.2.11](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.10...v0.2.11) (2026-05-24)

### Documentation

- **changelog:** add note explaining the 2026-05-24 git history rewrite ([ceff1a3](https://github.com/NorthlandPositronics/Cogtrix/commit/ceff1a333cece39fc1c790dc70d4809758d61d9a))
- **changelog:** add note explaining the 2026-05-24 git history rewrite ([648075b](https://github.com/NorthlandPositronics/Cogtrix/commit/648075b0f82e4ad22a011e0297d11798ef61100c))

## [0.2.10](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.9...v0.2.10) (2026-05-18)

### Features

- **gate2:** multi-turn scenario foundation — schema + runner loop ([#1538](https://github.com/NorthlandPositronics/Cogtrix/issues/1538)) ([3df419e](https://github.com/NorthlandPositronics/Cogtrix/commit/3df419e51eada539d0adb40d5788c3fd38ddf27f))
- **gate2:** multi-turn scenario foundation — schema + runner loop ([#1538](https://github.com/NorthlandPositronics/Cogtrix/issues/1538)) ([bba1488](https://github.com/NorthlandPositronics/Cogtrix/commit/bba1488deba968ee96f01ead86f2e093516e75e6))
- **gate2:** per-turn judge scoring with weighted aggregate ([#1545](https://github.com/NorthlandPositronics/Cogtrix/issues/1545)) ([6fa1c48](https://github.com/NorthlandPositronics/Cogtrix/commit/6fa1c4860d3cc0f0233d48cff0594d17e844a276))
- **gate2:** per-turn judge scoring with weighted aggregate ([#1545](https://github.com/NorthlandPositronics/Cogtrix/issues/1545)) ([c75f8ef](https://github.com/NorthlandPositronics/Cogtrix/commit/c75f8ef5782a24517650aacdab4cc11946804ad8))
- **saml:** add assertion replay protection via nonce cache (closes [#1135](https://github.com/NorthlandPositronics/Cogtrix/issues/1135)) ([dcd026a](https://github.com/NorthlandPositronics/Cogtrix/commit/dcd026ac77932248e0220942eb1c0077e9eafc89))

### Bug Fixes

- **api:** add ENABLE_ORG_SCOPING feature flag default false, fix test_campaigns.py CI failure ([c95db7f](https://github.com/NorthlandPositronics/Cogtrix/commit/c95db7fd10c6cb300c7a3a42c9a91b10d2e0b438)), closes [#1136](https://github.com/NorthlandPositronics/Cogtrix/issues/1136)
- **api:** address TL review findings on PR [#1536](https://github.com/NorthlandPositronics/Cogtrix/issues/1536) ([4ca0541](https://github.com/NorthlandPositronics/Cogtrix/commit/4ca0541bc5abbe5f5c99d6c2b86182620cc2f1b5)), closes [#1136](https://github.com/NorthlandPositronics/Cogtrix/issues/1136)
- **api:** timeout wrapping for bare llm.invoke() in config wizard ([#1568](https://github.com/NorthlandPositronics/Cogtrix/issues/1568)) ([945d0ca](https://github.com/NorthlandPositronics/Cogtrix/commit/945d0ca962dc590c2bebb1d247e3d4662aad670a))
- **api:** timeout wrapping for bare llm.invoke() in config wizard ([#1568](https://github.com/NorthlandPositronics/Cogtrix/issues/1568)) ([0a0c033](https://github.com/NorthlandPositronics/Cogtrix/commit/0a0c033e7327fc35f4624edef50d5ed1b88370ea))
- **assistant/handler:** add timeout=30 to PR validation subprocess.run ([#1535](https://github.com/NorthlandPositronics/Cogtrix/issues/1535)) ([383ce0f](https://github.com/NorthlandPositronics/Cogtrix/commit/383ce0f0d9e16b5ae2f13a4624d00e8e4c01bc92))
- **assistant/handler:** add timeout=30 to PR validation subprocess.run ([#1535](https://github.com/NorthlandPositronics/Cogtrix/issues/1535)) ([58dd3f9](https://github.com/NorthlandPositronics/Cogtrix/commit/58dd3f90696da2ef5a53afea1bae21ae2b549266))
- **assistant/session:** add hard overflow cap to prevent unbounded session growth (closes [#1075](https://github.com/NorthlandPositronics/Cogtrix/issues/1075)) ([f943bff](https://github.com/NorthlandPositronics/Cogtrix/commit/f943bff53f695c75d51c562499d0aee38e1bc7c9))
- **assistant/session:** add hard overflow cap to prevent unbounded session growth (closes [#1075](https://github.com/NorthlandPositronics/Cogtrix/issues/1075)) ([f943bff](https://github.com/NorthlandPositronics/Cogtrix/commit/f943bff53f695c75d51c562499d0aee38e1bc7c9))
- **assistant/session:** add hard overflow cap to prevent unbounded session growth (closes [#1075](https://github.com/NorthlandPositronics/Cogtrix/issues/1075)) ([c6c8173](https://github.com/NorthlandPositronics/Cogtrix/commit/c6c817378451cec071bbdd637c518e23e489f7e7))
- **assistant:** add timeout to \_extract_facts() to prevent knowledge extraction pool exhaustion ([#1140](https://github.com/NorthlandPositronics/Cogtrix/issues/1140)) ([55f3228](https://github.com/NorthlandPositronics/Cogtrix/commit/55f3228606b764d96dd9061ed43336b77fb9b577))
- **assistant:** add timeout to \_extract_facts() to prevent knowledge extraction pool exhaustion ([#1140](https://github.com/NorthlandPositronics/Cogtrix/issues/1140)) ([8dc293d](https://github.com/NorthlandPositronics/Cogtrix/commit/8dc293d4a1e7fe9573c74d9fca5c8524aad245e8))
- **assistant:** move prerecord_user inside session lock to prevent race condition ([e3e86fa](https://github.com/NorthlandPositronics/Cogtrix/commit/e3e86fac1723818d481336f76ab640569e86a9a5))
- **assistant:** move prerecord_user inside session lock to prevent race condition ([#905](https://github.com/NorthlandPositronics/Cogtrix/issues/905)) ([df0013b](https://github.com/NorthlandPositronics/Cogtrix/commit/df0013b85cff6837136a87fcf110b176029d4c43))
- **assistant:** move prerecord_user inside session lock to prevent race condition ([#905](https://github.com/NorthlandPositronics/Cogtrix/issues/905)) ([df0013b](https://github.com/NorthlandPositronics/Cogtrix/commit/df0013b85cff6837136a87fcf110b176029d4c43))
- **assistant:** shut down ThreadPoolExecutor on **init** failure (closes [#908](https://github.com/NorthlandPositronics/Cogtrix/issues/908)) ([7e3f20d](https://github.com/NorthlandPositronics/Cogtrix/commit/7e3f20d093086f52f425f31ea258db2648c05976))
- **assistant:** shut down ThreadPoolExecutor on **init** failure (closes [#908](https://github.com/NorthlandPositronics/Cogtrix/issues/908)) ([7e3f20d](https://github.com/NorthlandPositronics/Cogtrix/commit/7e3f20d093086f52f425f31ea258db2648c05976))
- **assistant:** shut down ThreadPoolExecutor on **init** failure (closes [#908](https://github.com/NorthlandPositronics/Cogtrix/issues/908)) ([1a4d3da](https://github.com/NorthlandPositronics/Cogtrix/commit/1a4d3dae3c52ac121c2d9548ab9b454ba7f011dd))
- **assistant:** wrap LLMJudge.classify() llm.invoke() with ThreadPoolExecutor timeout ([#1122](https://github.com/NorthlandPositronics/Cogtrix/issues/1122)) ([d253e36](https://github.com/NorthlandPositronics/Cogtrix/commit/d253e3644f4d20dbcd360f10d8c8bd5392f5ce4d))
- **assistant:** wrap LLMJudge.classify() llm.invoke() with ThreadPoolExecutor timeout ([#1122](https://github.com/NorthlandPositronics/Cogtrix/issues/1122)) ([37f0454](https://github.com/NorthlandPositronics/Cogtrix/commit/37f0454aa25623def4477b5262f052bda1f841d1))
- correct TOCTOU mitigation — check is_symlink() before resolve() (issue [#808](https://github.com/NorthlandPositronics/Cogtrix/issues/808)) ([295781b](https://github.com/NorthlandPositronics/Cogtrix/commit/295781bdb7d279e7b25d7046dae13d20b70d1e58))
- **docs:** add 7 missing tool sections to TOOLS_REFERENCE.md, fix Slack ordering ([2a02abd](https://github.com/NorthlandPositronics/Cogtrix/commit/2a02abde495148fa659b3476708c9a9f2f271f6a))
- **docs:** add 7 missing tool sections to TOOLS_REFERENCE.md, fix Slack ordering ([c10e063](https://github.com/NorthlandPositronics/Cogtrix/commit/c10e063866b828cf28a4c3f88eb2e4d2cfb87b57)), closes [#1298](https://github.com/NorthlandPositronics/Cogtrix/issues/1298)
- **docs:** correct docker-compose reference, add missing slash commands to ARCHITECTURE.md ([cdac3d1](https://github.com/NorthlandPositronics/Cogtrix/commit/cdac3d197bb4ba347b63d319393a1e4ba6d77a65))
- **docs:** correct docker-compose reference, add missing slash commands to ARCHITECTURE.md ([9df3076](https://github.com/NorthlandPositronics/Cogtrix/commit/9df3076a7ed0ec63f02f605cf0b614a890730db4))
- **docs:** remove duplicate Slack Messaging TOC entry (Lena Corbin QA) ([0c8753e](https://github.com/NorthlandPositronics/Cogtrix/commit/0c8753e27e6faf464f125c199770f6383499961d))
- **eval:** revert deepseek-v4-flash from deepseek-reasoner to actual v4-flash model ([40c2eba](https://github.com/NorthlandPositronics/Cogtrix/commit/40c2ebafdeb6b29602a7b8f86af75f6469c42f59))
- **eval:** revert deepseek-v4-flash from deepseek-reasoner to actual v4-flash model ([a5c66d9](https://github.com/NorthlandPositronics/Cogtrix/commit/a5c66d94aa2b8085cefd8bbebe84cf2d1ccd5d90))
- **memory:** guard distill_summary against llm=None (closes [#1082](https://github.com/NorthlandPositronics/Cogtrix/issues/1082)) ([855c3cf](https://github.com/NorthlandPositronics/Cogtrix/commit/855c3cf98a4e2a3e082b2bf27b5bbdfc0e7717ea))
- **memory:** guard distill_summary against llm=None (closes [#1082](https://github.com/NorthlandPositronics/Cogtrix/issues/1082)) ([855c3cf](https://github.com/NorthlandPositronics/Cogtrix/commit/855c3cf98a4e2a3e082b2bf27b5bbdfc0e7717ea))
- **orchestration:** add effort gate to no-checkpoint thinking break to prevent lazy refusal ([#1520](https://github.com/NorthlandPositronics/Cogtrix/issues/1520)) ([b6a5a8f](https://github.com/NorthlandPositronics/Cogtrix/commit/b6a5a8f45362a0b3fc860720ae364afd04152a16))
- **orchestration:** add effort gate to no-checkpoint thinking break to prevent lazy refusal (closes [#1520](https://github.com/NorthlandPositronics/Cogtrix/issues/1520)) ([78b26d9](https://github.com/NorthlandPositronics/Cogtrix/commit/78b26d972f6456ad5f55d088329f391a144b3a0b))
- **orchestration:** address PR [#1525](https://github.com/NorthlandPositronics/Cogtrix/issues/1525) review blockers on effort gate ([b3af85e](https://github.com/NorthlandPositronics/Cogtrix/commit/b3af85e990b69a44950d3dfa75772b09a9ade0d2))
- **orchestration:** anti-fabrication clause covers URLs; read_file steers to http_get ([#1532](https://github.com/NorthlandPositronics/Cogtrix/issues/1532)) ([a02be42](https://github.com/NorthlandPositronics/Cogtrix/commit/a02be420d659f6085260cefecf259148a05a59c8))
- **orchestration:** anti-fabrication clause covers URLs; read_file steers to http_get ([#1532](https://github.com/NorthlandPositronics/Cogtrix/issues/1532)) ([484f7d1](https://github.com/NorthlandPositronics/Cogtrix/commit/484f7d1319e7f7813253cc849e49ee55ca1ba4ea))
- **orchestration:** apply enhanced anti-fabrication clause from [#1516](https://github.com/NorthlandPositronics/Cogtrix/issues/1516) into thinking-break prompt ([9f8a5cd](https://github.com/NorthlandPositronics/Cogtrix/commit/9f8a5cd6699c7e6e5bb27e4356aad3c962a16693))
- **orchestration:** conditionally pass \_cogtrix_disable_retries to avoid leaking to raw models ([dcf14ae](https://github.com/NorthlandPositronics/Cogtrix/commit/dcf14aebe4355c6c49149af05b4290834bd2b173))
- **orchestration:** per-invoke timeout wrapping for thread_llm.invoke() in deep_think.py ([#1569](https://github.com/NorthlandPositronics/Cogtrix/issues/1569)) ([3238474](https://github.com/NorthlandPositronics/Cogtrix/commit/323847437b3de2c0c09592fa8904bb175903967e))
- **orchestration:** per-invoke timeout wrapping for thread_llm.invoke() in deep_think.py ([#1569](https://github.com/NorthlandPositronics/Cogtrix/issues/1569)) ([925a0e3](https://github.com/NorthlandPositronics/Cogtrix/commit/925a0e3d771c4aa4859163ca8577e3d28a0b485a))
- **orchestration:** promote checkpoint nudge to SystemMessage for higher salience ([#1532](https://github.com/NorthlandPositronics/Cogtrix/issues/1532) Bug 4) ([4822982](https://github.com/NorthlandPositronics/Cogtrix/commit/48229827c34752693e2e0822a3633298a116c4b7))
- **orchestration:** promote checkpoint nudge to SystemMessage for higher salience ([#1532](https://github.com/NorthlandPositronics/Cogtrix/issues/1532) Bug 4) ([4822982](https://github.com/NorthlandPositronics/Cogtrix/commit/48229827c34752693e2e0822a3633298a116c4b7))
- **orchestration:** promote checkpoint nudge to SystemMessage for higher salience ([#1532](https://github.com/NorthlandPositronics/Cogtrix/issues/1532) Bug 4) ([b3b0a85](https://github.com/NorthlandPositronics/Cogtrix/commit/b3b0a8561ffc0154ccfdb9c1b61069c1674ced8a))
- **orchestration:** sanitize tool names in ToolMessage error content ([#1070](https://github.com/NorthlandPositronics/Cogtrix/issues/1070)) ([ebca521](https://github.com/NorthlandPositronics/Cogtrix/commit/ebca5215d530afe6767bc277da644bf3a23dbe53))
- **orchestration:** sanitize tool names in ToolMessage error content ([#1070](https://github.com/NorthlandPositronics/Cogtrix/issues/1070)) ([ebca521](https://github.com/NorthlandPositronics/Cogtrix/commit/ebca5215d530afe6767bc277da644bf3a23dbe53))
- **orchestration:** sanitize tool names in ToolMessage error content ([#1070](https://github.com/NorthlandPositronics/Cogtrix/issues/1070)) ([dedce84](https://github.com/NorthlandPositronics/Cogtrix/commit/dedce84197cd11daa38fe0bdfe12626320ff836d))
- **orchestration:** sanitize tool names in ToolMessage error content (closes [#1070](https://github.com/NorthlandPositronics/Cogtrix/issues/1070)) ([49ab1dd](https://github.com/NorthlandPositronics/Cogtrix/commit/49ab1dd755ca050611e4776f577adaacda966e29))
- **orchestration:** sanitize tool names in ToolMessage error content (closes [#1070](https://github.com/NorthlandPositronics/Cogtrix/issues/1070)) ([49ab1dd](https://github.com/NorthlandPositronics/Cogtrix/commit/49ab1dd755ca050611e4776f577adaacda966e29))
- **orchestration:** sanitize tool names in ToolMessage error content (closes [#1070](https://github.com/NorthlandPositronics/Cogtrix/issues/1070)) ([72228f6](https://github.com/NorthlandPositronics/Cogtrix/commit/72228f6f87373016f7901e862585c85edf484f86))
- **orchestration:** scope \_drain_background_compression_jobs to target cache (closes [#901](https://github.com/NorthlandPositronics/Cogtrix/issues/901)) ([#1005](https://github.com/NorthlandPositronics/Cogtrix/issues/1005)) ([#1056](https://github.com/NorthlandPositronics/Cogtrix/issues/1056))
  ([5970120](https://github.com/NorthlandPositronics/Cogtrix/commit/5970120a7001e274644cd7da7265cae43c1bfec7))
- **orchestration:** scope effort gate to current turn only ([#1532](https://github.com/NorthlandPositronics/Cogtrix/issues/1532)) ([e91600d](https://github.com/NorthlandPositronics/Cogtrix/commit/e91600d92b56e426f56f9697827111a6eb5ee2bc))
- **orchestration:** scope effort gate to current turn only ([#1532](https://github.com/NorthlandPositronics/Cogtrix/issues/1532)) ([5c52212](https://github.com/NorthlandPositronics/Cogtrix/commit/5c522125019b6c32a73368d1b412dc27d7be6044))
- **orchestration:** suppress thinking-break arm on stub-only rounds and prevent fabrication ([#1510](https://github.com/NorthlandPositronics/Cogtrix/issues/1510)) ([9aefcd4](https://github.com/NorthlandPositronics/Cogtrix/commit/9aefcd4bcf21538577c0f0a004ccd4b47f82d62b))
- **orchestration:** suppress thinking-break arm on stub-only rounds and prevent fabrication ([#1510](https://github.com/NorthlandPositronics/Cogtrix/issues/1510)) ([f5a6266](https://github.com/NorthlandPositronics/Cogtrix/commit/f5a62661b8cec78edde696516e1da813394c1456))
- **orchestration:** wrap force_delegation llm.invoke with ThreadPoolExecutor timeout ([#1164](https://github.com/NorthlandPositronics/Cogtrix/issues/1164)) ([8b097ff](https://github.com/NorthlandPositronics/Cogtrix/commit/8b097ffd883b59ee73f5206a2dc74abd6fc93bd6))
- **orchestration:** wrap force_delegation llm.invoke with ThreadPoolExecutor timeout ([#1164](https://github.com/NorthlandPositronics/Cogtrix/issues/1164)) ([8b097ff](https://github.com/NorthlandPositronics/Cogtrix/commit/8b097ffd883b59ee73f5206a2dc74abd6fc93bd6))
- **orchestration:** wrap force_delegation llm.invoke with ThreadPoolExecutor timeout ([#1164](https://github.com/NorthlandPositronics/Cogtrix/issues/1164)) ([a5a6de8](https://github.com/NorthlandPositronics/Cogtrix/commit/a5a6de845e93e43a529e0cf161dd0dae74e719cd))
- **orchestration:** wrap reflection_delegate \_call_llm with ThreadPoolExecutor timeout ([#1558](https://github.com/NorthlandPositronics/Cogtrix/issues/1558)) ([a71638e](https://github.com/NorthlandPositronics/Cogtrix/commit/a71638ea02adf5bbf9e88a6dc5ca3b33e9187b7e))
- **orchestration:** wrap reflection_delegate \_call_llm with ThreadPoolExecutor timeout ([#1558](https://github.com/NorthlandPositronics/Cogtrix/issues/1558)) ([a71638e](https://github.com/NorthlandPositronics/Cogtrix/commit/a71638ea02adf5bbf9e88a6dc5ca3b33e9187b7e))
- **orchestration:** wrap reflection_delegate \_call_llm with ThreadPoolExecutor timeout ([#1558](https://github.com/NorthlandPositronics/Cogtrix/issues/1558)) ([37b326b](https://github.com/NorthlandPositronics/Cogtrix/commit/37b326b9e90765dd80c3be7ffaa15ad3b1a69aa6))
- **prompt:** add ThreadPoolExecutor 60s timeout to optimize_prompt() llm.invoke() ([3b478e2](https://github.com/NorthlandPositronics/Cogtrix/commit/3b478e257234e301d568aef38ceeb851358a740e))
- **prompt:** add ThreadPoolExecutor 60s timeout to optimize_prompt() llm.invoke() ([c27d100](https://github.com/NorthlandPositronics/Cogtrix/commit/c27d1001c125cae01be50a346be92ddb1b60a0ad))
- **providers:** add refresh_token and client_secret to \_redact_url sensitive_keys ([#1508](https://github.com/NorthlandPositronics/Cogtrix/issues/1508)) ([d2b09f7](https://github.com/NorthlandPositronics/Cogtrix/commit/d2b09f7068466e90e702f4f4dd042d0e98899ef9))
- **providers:** address CODEOWNERS review on PR [#1553](https://github.com/NorthlandPositronics/Cogtrix/issues/1553) ([acf5fbe](https://github.com/NorthlandPositronics/Cogtrix/commit/acf5fbe17cb75f5edcec60ee1430a142091d21d8))
- **providers:** apply computed exponential backoff delay in RetryableChatModel (closes [#1511](https://github.com/NorthlandPositronics/Cogtrix/issues/1511)) ([85ec8ac](https://github.com/NorthlandPositronics/Cogtrix/commit/85ec8acfd968d0eb9cf05c280f9807199e329995))
- **providers:** apply computed exponential backoff delay in RetryableChatModel (closes [#1511](https://github.com/NorthlandPositronics/Cogtrix/issues/1511)) ([aabf695](https://github.com/NorthlandPositronics/Cogtrix/commit/aabf695315e2c5b2dfe4deb6f368bcbe55d618cf))
- **providers:** bind_tools must re-wrap result in RetryableChatModel to prevent \_cogtrix_disable_retries leak ([15dec77](https://github.com/NorthlandPositronics/Cogtrix/commit/15dec77e4f63859b719e08093ae04d8080f40cfb))
- **providers:** disable inner retry loop when invoked via executor to prevent thread pool starvation ([#1069](https://github.com/NorthlandPositronics/Cogtrix/issues/1069)) ([e3669fd](https://github.com/NorthlandPositronics/Cogtrix/commit/e3669fd7b531e393aab3a57e7eacf0bc596ba76f))
- **providers:** disable inner retry loop when invoked via executor to prevent thread pool starvation (closes [#1069](https://github.com/NorthlandPositronics/Cogtrix/issues/1069)) ([93a00b0](https://github.com/NorthlandPositronics/Cogtrix/commit/93a00b0c014cd71fc2c8e39240752bd76d28e5e0))
- **providers:** expand \_redact_url sensitive_keys to cover common credential params (closes [#1071](https://github.com/NorthlandPositronics/Cogtrix/issues/1071), [#1508](https://github.com/NorthlandPositronics/Cogtrix/issues/1508)) ([a5cf386](https://github.com/NorthlandPositronics/Cogtrix/commit/a5cf386ae2f7be31717767a2ef25f04adc3038e0))
- **providers:** redact DeepSeek base_url in warning logs ([#1106](https://github.com/NorthlandPositronics/Cogtrix/issues/1106)) ([8707e5d](https://github.com/NorthlandPositronics/Cogtrix/commit/8707e5d788971c157539879261d27e350a146c9d))
- **providers:** redact DeepSeek base_url in warning logs ([#1106](https://github.com/NorthlandPositronics/Cogtrix/issues/1106)) ([17e1ce7](https://github.com/NorthlandPositronics/Cogtrix/commit/17e1ce7631aae75f9178e99bfe38631839bdf5d6))
- **providers:** remove dead \_wrapped_attrs computation in RetryableChatModel ([feebeaf](https://github.com/NorthlandPositronics/Cogtrix/commit/feebeaf416caa10fe68a2a29e64637bd39d1d40b))
- **providers:** remove dead \_wrapped_attrs computation in RetryableChatModel ([#1526](https://github.com/NorthlandPositronics/Cogtrix/issues/1526)) ([1210d9c](https://github.com/NorthlandPositronics/Cogtrix/commit/1210d9c8d1ed7cac6768473379110602a90cb476)), closes [#1083](https://github.com/NorthlandPositronics/Cogtrix/issues/1083)
- **saml:** add assertion replay protection via nonce cache (closes [#1135](https://github.com/NorthlandPositronics/Cogtrix/issues/1135)) ([de01b19](https://github.com/NorthlandPositronics/Cogtrix/commit/de01b197540b7550f7a1a00bb6e0470ef1c237d0))
- **saml:** add assertion_id guard and periodic nonce cleanup (closes [#1135](https://github.com/NorthlandPositronics/Cogtrix/issues/1135)) ([447adcb](https://github.com/NorthlandPositronics/Cogtrix/commit/447adcb60cc3b22220046c24f25280aca88cafed))
- **security:** tool name sanitization + TOCTOU path validation closure ([9d1a539](https://github.com/NorthlandPositronics/Cogtrix/commit/9d1a5390f4c86395712e9ceaedd943876aa7ceed))
- **security:** tool name sanitization + TOCTOU path validation closure ([#1070](https://github.com/NorthlandPositronics/Cogtrix/issues/1070), [#924](https://github.com/NorthlandPositronics/Cogtrix/issues/924)) ([9d1a539](https://github.com/NorthlandPositronics/Cogtrix/commit/9d1a5390f4c86395712e9ceaedd943876aa7ceed))
- **setup_wizard:** resolve ruff E402 and B904 lint errors ([36a4d2f](https://github.com/NorthlandPositronics/Cogtrix/commit/36a4d2f39536a0a9df7d1afc5f960af7b98902c1))
- **setup:** wrap 4 bare llm.invoke() calls in setup_wizard.py with ThreadPoolExecutor timeout ([#1567](https://github.com/NorthlandPositronics/Cogtrix/issues/1567)) ([80b19f2](https://github.com/NorthlandPositronics/Cogtrix/commit/80b19f2e4f35e4c74e7894eb558b86c4ef747b5c))
- **setup:** wrap 4 bare llm.invoke() calls in setup_wizard.py with ThreadPoolExecutor timeout ([#1567](https://github.com/NorthlandPositronics/Cogtrix/issues/1567)) ([3c3122a](https://github.com/NorthlandPositronics/Cogtrix/commit/3c3122adf04e3582ac9c45065116ba4f6e79bfc9))
- **test_file_ops:** apply black formatting to regression test ([b41b115](https://github.com/NorthlandPositronics/Cogtrix/commit/b41b11524665ebcb31397d3c6535bf8b445ef545))
- **tests:** apply black formatting to test_bugfix_audit_regression.py ([0c0afc4](https://github.com/NorthlandPositronics/Cogtrix/commit/0c0afc434436745cfa19ce5ca0f422ced96cbc2c))
- **tests:** remove xfail from test_all_subprocess_calls_have_timeout ([711fbc2](https://github.com/NorthlandPositronics/Cogtrix/commit/711fbc22a7220b32789a247c36788448a68ef1fd))
- **tests:** remove xfail from TestBug906SubprocessTimeout now that fix is merged ([ec35490](https://github.com/NorthlandPositronics/Cogtrix/commit/ec35490473bb68760d488be8a3470e8377e02fd1))
- **tests:** update import from guardrails to \_security_patterns ([8ee7d82](https://github.com/NorthlandPositronics/Cogtrix/commit/8ee7d82194beefa2c1744781b7c88da58bf66852))
- **tools/configure:** log warning on transitive ImportError, stay silent on ModuleNotFoundError (closes [#1089](https://github.com/NorthlandPositronics/Cogtrix/issues/1089)) ([3b58e12](https://github.com/NorthlandPositronics/Cogtrix/commit/3b58e1206e0a0d2f2a8329b71bb398e9e3359d43))
- **tools/configure:** log warning on transitive ImportError, stay silent on ModuleNotFoundError (closes [#1089](https://github.com/NorthlandPositronics/Cogtrix/issues/1089)) ([3b58e12](https://github.com/NorthlandPositronics/Cogtrix/commit/3b58e1206e0a0d2f2a8329b71bb398e9e3359d43))
- **tools/delegate:** expand \_DELEGATE_EXCLUDED_TOOLS with 15 missing tools (closes [#1072](https://github.com/NorthlandPositronics/Cogtrix/issues/1072)) ([7f5c480](https://github.com/NorthlandPositronics/Cogtrix/commit/7f5c4801bcbffeddb8f3a8d3e1178d6494e66d19))
- **tools/delegate:** expand \_DELEGATE_EXCLUDED_TOOLS with 15 missing tools (closes [#1072](https://github.com/NorthlandPositronics/Cogtrix/issues/1072)) ([32a4820](https://github.com/NorthlandPositronics/Cogtrix/commit/32a4820dc273b22937bc5c82f0fedf2f7b8a6f15))
- **tools/file_ops:** close TOCTOU window in \_validate_path (closes [#924](https://github.com/NorthlandPositronics/Cogtrix/issues/924)) ([c4dc9e5](https://github.com/NorthlandPositronics/Cogtrix/commit/c4dc9e53cd925b5fbf927a7b27ba679a879ab326))
- **tools/file_ops:** fix error message to say 'patchable' not 'readable' (closes [#967](https://github.com/NorthlandPositronics/Cogtrix/issues/967)) ([14f1079](https://github.com/NorthlandPositronics/Cogtrix/commit/14f1079b5562f7db31c3040f32ebd5b8078d4b56))
- **tools/file_ops:** fix error message to say 'patchable' not 'readable' (closes [#967](https://github.com/NorthlandPositronics/Cogtrix/issues/967)) ([384ce38](https://github.com/NorthlandPositronics/Cogtrix/commit/384ce38f84ba1f8f19bac602a56282118f81d552))
- **tools/github:** wire \_classify_gh_error into 4 error paths (closes [#1453](https://github.com/NorthlandPositronics/Cogtrix/issues/1453)) ([ff3b92e](https://github.com/NorthlandPositronics/Cogtrix/commit/ff3b92ef934716b771479cadfa0894e07f13f02d))
- **tools/github:** wire \_classify_gh_error into 4 error paths (closes [#1453](https://github.com/NorthlandPositronics/Cogtrix/issues/1453)) ([3e473e2](https://github.com/NorthlandPositronics/Cogtrix/commit/3e473e2c73b9138e49868e6b9bf41808daf444b3))
- **tools:** add file locking to agent_messaging to prevent lost-update race ([#973](https://github.com/NorthlandPositronics/Cogtrix/issues/973), [#977](https://github.com/NorthlandPositronics/Cogtrix/issues/977)) ([b2e314f](https://github.com/NorthlandPositronics/Cogtrix/commit/b2e314f301b0406213d2054238c278c33c0f81fa))
- **tools:** add timeout to all four github_tools subprocess.run calls ([b84c476](https://github.com/NorthlandPositronics/Cogtrix/commit/b84c4766307507279a5a359ed7c92777995d956a))
- **tools:** add timeout to all four github_tools subprocess.run calls ([2676b62](https://github.com/NorthlandPositronics/Cogtrix/commit/2676b62903fa32345790db01d8447927cc3aee2e))
- **tools:** file locking for agent_messaging lost-update race ([#973](https://github.com/NorthlandPositronics/Cogtrix/issues/973), [#977](https://github.com/NorthlandPositronics/Cogtrix/issues/977)) ([a2a1351](https://github.com/NorthlandPositronics/Cogtrix/commit/a2a135118d38141658cd6846138ee6ce23636f3b))
- **tools:** file locking for agent_messaging lost-update race ([#973](https://github.com/NorthlandPositronics/Cogtrix/issues/973), [#977](https://github.com/NorthlandPositronics/Cogtrix/issues/977)) ([#1528](https://github.com/NorthlandPositronics/Cogtrix/issues/1528)) ([a2a1351](https://github.com/NorthlandPositronics/Cogtrix/commit/a2a135118d38141658cd6846138ee6ce23636f3b))
- **tools:** prevent list_directory from following symlinks in glob (closes [#944](https://github.com/NorthlandPositronics/Cogtrix/issues/944)) ([#1055](https://github.com/NorthlandPositronics/Cogtrix/issues/1055)) ([636bb1e](https://github.com/NorthlandPositronics/Cogtrix/commit/636bb1e45ee33ee4d226a026daa24dd1b086e06b))
- **tools:** redact secrets in LLM prompts for self_improve and generate_tests ([#1176](https://github.com/NorthlandPositronics/Cogtrix/issues/1176)) ([128296e](https://github.com/NorthlandPositronics/Cogtrix/commit/128296ed5886a7a0846416ab23f209c55753aefe))
- **tools:** redact secrets in LLM prompts for self_improve and generate_tests ([#1176](https://github.com/NorthlandPositronics/Cogtrix/issues/1176)) ([6ade067](https://github.com/NorthlandPositronics/Cogtrix/commit/6ade067cc8277db9e2b87d0a7f26eba2f1c37f26))
- **tools:** remove naive ".." substring check from \_validate_path ([#927](https://github.com/NorthlandPositronics/Cogtrix/issues/927)) ([1974c8e](https://github.com/NorthlandPositronics/Cogtrix/commit/1974c8ea1a83d07e99190ebf5d4fc7378672e281))
- **tools:** remove naive ".." substring check from \_validate_path ([#927](https://github.com/NorthlandPositronics/Cogtrix/issues/927)) ([1974c8e](https://github.com/NorthlandPositronics/Cogtrix/commit/1974c8ea1a83d07e99190ebf5d4fc7378672e281))
- **tools:** remove naive ".." substring check from \_validate_path ([#927](https://github.com/NorthlandPositronics/Cogtrix/issues/927)) ([1d1fe97](https://github.com/NorthlandPositronics/Cogtrix/commit/1d1fe97e67c347c57c6b465181de163d57999e7f))

### Documentation

- align docs with current codebase (closes [#1504](https://github.com/NorthlandPositronics/Cogtrix/issues/1504)) ([24fb83f](https://github.com/NorthlandPositronics/Cogtrix/commit/24fb83f9d1c990d441945024fcdce1bbf14234da))
- align docs with current codebase (closes [#1504](https://github.com/NorthlandPositronics/Cogtrix/issues/1504)) ([b8d81be](https://github.com/NorthlandPositronics/Cogtrix/commit/b8d81be92c5799c57705a9fc21c01f2db019900c))
- **readme:** complete rewrite — lead with what it does, not what it claims ([c769f93](https://github.com/NorthlandPositronics/Cogtrix/commit/c769f93583eb3fcd4eea80810fb90f8eeb3580b2))
- **readme:** rework examples for finance + ML/AI student angle ([2e9c1f5](https://github.com/NorthlandPositronics/Cogtrix/commit/2e9c1f545c65fdaf2edefe38e5da5a838db7ec75)), closes [#1504](https://github.com/NorthlandPositronics/Cogtrix/issues/1504)
- **readme:** wrap long WhatsApp-daemon paragraph for MD013 ([5b4f9fb](https://github.com/NorthlandPositronics/Cogtrix/commit/5b4f9fbfec1373ddfc21bf27d4df10c57627ec78)), closes [#1504](https://github.com/NorthlandPositronics/Cogtrix/issues/1504)

## [0.2.9](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.8...v0.2.9) (2026-05-18)

### Bug Fixes

- **agent/safety:** propagate tool_trust to dynamic tool loading (closes [#1000](https://github.com/NorthlandPositronics/Cogtrix/issues/1000)) ([73444ce](https://github.com/NorthlandPositronics/Cogtrix/commit/73444ceaf2661e8ce36e7692022c8b1083532325))
- **api/mcp:** offload restart_server to asyncio.to_thread ([#1198](https://github.com/NorthlandPositronics/Cogtrix/issues/1198)) ([f1307f4](https://github.com/NorthlandPositronics/Cogtrix/commit/f1307f4d71059574ce3ed6bea9f479f74c1d953c))
- **assistant/campaign:** re-check target status under lock before follow-up dispatch ([#1488](https://github.com/NorthlandPositronics/Cogtrix/issues/1488)) ([25a2d1a](https://github.com/NorthlandPositronics/Cogtrix/commit/25a2d1a636b6a6e96b71100601b1b15661d86fcb))
- **assistant/campaign:** re-check target status under lock before follow-up dispatch (closes [#1121](https://github.com/NorthlandPositronics/Cogtrix/issues/1121)) ([25a2d1a](https://github.com/NorthlandPositronics/Cogtrix/commit/25a2d1a636b6a6e96b71100601b1b15661d86fcb))
- **assistant/campaign:** re-check target status under lock before follow-up dispatch (closes [#1121](https://github.com/NorthlandPositronics/Cogtrix/issues/1121)) ([3b409aa](https://github.com/NorthlandPositronics/Cogtrix/commit/3b409aa4a24905fe31a237fc78811598a9c8a6d7))
- **assistant/scheduler:** prevent double-start race in MessageScheduler.start() ([47c7260](https://github.com/NorthlandPositronics/Cogtrix/commit/47c72605d8d6865ed5aa773c7b9b7b932bf5b268)), closes [#1120](https://github.com/NorthlandPositronics/Cogtrix/issues/1120)
- **assistant/scheduler:** prevent double-start race under lock (closes [#1120](https://github.com/NorthlandPositronics/Cogtrix/issues/1120)) ([6f65dbf](https://github.com/NorthlandPositronics/Cogtrix/commit/6f65dbf342884966782cb16c841758d42abafda5))
- **assistant:** add exponential backoff to \_dispatch_loop for persistent errors ([#1077](https://github.com/NorthlandPositronics/Cogtrix/issues/1077)) ([7885dff](https://github.com/NorthlandPositronics/Cogtrix/commit/7885dffefbd094cc67e25fc1fa62718aa3ba8a80))
- **assistant:** add exponential backoff to \_dispatch_loop for persistent errors ([#1077](https://github.com/NorthlandPositronics/Cogtrix/issues/1077)) ([1ee1314](https://github.com/NorthlandPositronics/Cogtrix/commit/1ee131474f2e52ed7e570ae299e3b42f89d47f88))
- **memory/json_store:** bound \_session_locks with LRU eviction — prevents unbounded memory leak (closes [#1080](https://github.com/NorthlandPositronics/Cogtrix/issues/1080)) ([2b828eb](https://github.com/NorthlandPositronics/Cogtrix/commit/2b828eb365ec23b75b8c4d96765dbb80bce3e04a))
- **memory/json_store:** bound \_session_locks with LRU eviction (closes [#1080](https://github.com/NorthlandPositronics/Cogtrix/issues/1080)) ([895e657](https://github.com/NorthlandPositronics/Cogtrix/commit/895e6570c2bcfea3aeb1bdab36b198c17f1f6c7e))
- **memory:** wrap \_messages read in \_mode_lock during save() ([a8e025a](https://github.com/NorthlandPositronics/Cogtrix/commit/a8e025a075131fd629122b5f004838940df92cbc)), closes [#1496](https://github.com/NorthlandPositronics/Cogtrix/issues/1496)
- **memory:** wrap \_messages read in \_mode_lock during save() (closes [#1496](https://github.com/NorthlandPositronics/Cogtrix/issues/1496)) ([24623ba](https://github.com/NorthlandPositronics/Cogtrix/commit/24623ba28559aa8e1ff698013e22452869098d61))
- **orchestration:** re-raise UserCancelledRun in force_delegation and run_research_delegate ([220df95](https://github.com/NorthlandPositronics/Cogtrix/commit/220df957c939861b71f01884c0e43bfc434101c0))
- **orchestration:** re-raise UserCancelledRun in force_delegation and run_research_delegate ([a00249c](https://github.com/NorthlandPositronics/Cogtrix/commit/a00249c4d8fcd5093e392efdd3aab7f57cfe7d91))
- **tools:** add 5s guard timeout to proc.wait() after kill in shell.py ([#1202](https://github.com/NorthlandPositronics/Cogtrix/issues/1202)) ([a8e886d](https://github.com/NorthlandPositronics/Cogtrix/commit/a8e886dbe10aded3d99042d7ce4ed97003a4cd63))
- **tools:** add 5s guard timeout to proc.wait() after kill in shell.py ([#1202](https://github.com/NorthlandPositronics/Cogtrix/issues/1202)) ([770b1d1](https://github.com/NorthlandPositronics/Cogtrix/commit/770b1d1a7c1ce960e2c2552050fc6e5b75c9c112))
- **tools:** error sanitization follow-ups from architecture sweep ([#1453](https://github.com/NorthlandPositronics/Cogtrix/issues/1453), [#1454](https://github.com/NorthlandPositronics/Cogtrix/issues/1454)) ([3c50ae3](https://github.com/NorthlandPositronics/Cogtrix/commit/3c50ae3c089e5a28a6bab3d218802b207eff6889))
- **tools:** propagate UserCancelledRun through delegate_parallel (closes [#1173](https://github.com/NorthlandPositronics/Cogtrix/issues/1173)) ([a4c959e](https://github.com/NorthlandPositronics/Cogtrix/commit/a4c959e4512092767593179762c8dbcf92be4ef4))
- **tools:** sanitize googleapiclient HttpError — stop API key leak to LLM ([#1467](https://github.com/NorthlandPositronics/Cogtrix/issues/1467)) ([ba8fe46](https://github.com/NorthlandPositronics/Cogtrix/commit/ba8fe46d022361ce63fd53eb314df447d83fce9d))
- **tools:** sanitize googleapiclient HttpError — stop API key leak to LLM (closes [#1467](https://github.com/NorthlandPositronics/Cogtrix/issues/1467)) ([ba8fe46](https://github.com/NorthlandPositronics/Cogtrix/commit/ba8fe46d022361ce63fd53eb314df447d83fce9d))

## [0.2.8](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.7...v0.2.8) (2026-05-18)

### Features

- **ci:** reduce pipeline wall-clock time from 43min to ~15min ([c63198b](https://github.com/NorthlandPositronics/Cogtrix/commit/c63198b93863323a8d463f7d3e07d52253a7ef1f))
- **ci:** reduce pipeline wall-clock time from 43min to ~15min ([c0f3e6c](https://github.com/NorthlandPositronics/Cogtrix/commit/c0f3e6cae1dcb5c458425aa408047daf45bd528d)), closes [#1377](https://github.com/NorthlandPositronics/Cogtrix/issues/1377)
- **ci:** split unit test job into 3 parallel matrix shards ([133e8f5](https://github.com/NorthlandPositronics/Cogtrix/commit/133e8f5f5eaf3c9b7a8e4e639738c8d08dbb1cc5))
- **ci:** split unit test job into 3 parallel matrix shards ([ca59bcf](https://github.com/NorthlandPositronics/Cogtrix/commit/ca59bcf5596f8e837f150d96abec8bf6deff77bc)), closes [#1387](https://github.com/NorthlandPositronics/Cogtrix/issues/1387)
- **memory:** add test coverage for distillation.py ([155e134](https://github.com/NorthlandPositronics/Cogtrix/commit/155e1344190b300f854fa25bfa539e6c246586a8))
- **memory:** add test coverage for distillation.py ([c0a56e1](https://github.com/NorthlandPositronics/Cogtrix/commit/c0a56e127dfd534c9aed51f3ed111020532aa699))
- **orchestration:** detect and recover from incomplete multi-step actions ([25ba238](https://github.com/NorthlandPositronics/Cogtrix/commit/25ba2386e28e4dc43334a09324a048f9a2bbb4bf)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **tests:** add 18 tests for validate_password_complexity() ([37148b9](https://github.com/NorthlandPositronics/Cogtrix/commit/37148b98eb4e5d09af52287dc21af304e615f1f5))

### Bug Fixes

- **agent:** validate paths in \_compute_file_diff before reading files ([704fcd1](https://github.com/NorthlandPositronics/Cogtrix/commit/704fcd1768d3a612f1cc16d06aea0bd7c5fca500))
- **agent:** validate paths in \_compute_file_diff before reading files ([a4a8456](https://github.com/NorthlandPositronics/Cogtrix/commit/a4a845609e2527949a38da89b55d9f72efb49b11)), closes [#1003](https://github.com/NorthlandPositronics/Cogtrix/issues/1003)
- **api/memory:** acquire turn_lock before mutating memory state ([#1116](https://github.com/NorthlandPositronics/Cogtrix/issues/1116)) ([5ca2083](https://github.com/NorthlandPositronics/Cogtrix/commit/5ca2083d23a418f506078d865a03e626a660994f))
- **api/memory:** acquire turn_lock before mutating memory state (closes [#1116](https://github.com/NorthlandPositronics/Cogtrix/issues/1116)) ([fd988bb](https://github.com/NorthlandPositronics/Cogtrix/commit/fd988bbba351cba2002be0f2260ca7af33c1e5d7))
- **api/tasks:** enforce org-level isolation for task operations ([#1117](https://github.com/NorthlandPositronics/Cogtrix/issues/1117)) ([ba9d631](https://github.com/NorthlandPositronics/Cogtrix/commit/ba9d631799813273278fcb40f427feb50ab94b07))
- **api/tasks:** enforce org-level isolation for task operations (closes [#1117](https://github.com/NorthlandPositronics/Cogtrix/issues/1117)) ([593a5c6](https://github.com/NorthlandPositronics/Cogtrix/commit/593a5c638a6eaf1ede69cedcda1958c673288600))
- **api/ws:** wrap validator.validate() in asyncio.to_thread() to avoid event-loop block ([f500977](https://github.com/NorthlandPositronics/Cogtrix/commit/f500977c4055cd40c0c7430af61bdb9c1b65a956))
- **api/ws:** wrap validator.validate() in asyncio.to_thread() to avoid event-loop block ([384f0fd](https://github.com/NorthlandPositronics/Cogtrix/commit/384f0fde498079654b20b1402c8416fd95fc728a)), closes [#1084](https://github.com/NorthlandPositronics/Cogtrix/issues/1084)
- **assistant:** add sync fallback in \_send_follow_up when executor not started ([6a1feb3](https://github.com/NorthlandPositronics/Cogtrix/commit/6a1feb3ced626ee09fe4c942d05f86c8a7010d3a))
- **assistant:** add Unicode NFKC normalization + homoglyph folding to campaign sanitizer ([#1452](https://github.com/NorthlandPositronics/Cogtrix/issues/1452)) ([dc51ca4](https://github.com/NorthlandPositronics/Cogtrix/commit/dc51ca410ea7ecea148633d25318c56c034e09f3))
- **assistant:** add Unicode NFKC normalization + homoglyph folding to campaign sanitizer ([#1452](https://github.com/NorthlandPositronics/Cogtrix/issues/1452)) ([4833eb2](https://github.com/NorthlandPositronics/Cogtrix/commit/4833eb2bbbb2e435098ee79000122663e725c7b2))
- **assistant:** handle None input in InputGuard.check ([#825](https://github.com/NorthlandPositronics/Cogtrix/issues/825)) ([32ded75](https://github.com/NorthlandPositronics/Cogtrix/commit/32ded751f35bcbbd7162843fa0c2d287e06bc192))
- **assistant:** handle None input in InputGuard.check ([#825](https://github.com/NorthlandPositronics/Cogtrix/issues/825)) ([8c294fa](https://github.com/NorthlandPositronics/Cogtrix/commit/8c294fa48b691ac25fbd2b365b70267e6cfc0f82))
- **assistant:** raise ValueError when guardrails.enabled=false (closes [#1408](https://github.com/NorthlandPositronics/Cogtrix/issues/1408) Vector 3) ([d6ba1c5](https://github.com/NorthlandPositronics/Cogtrix/commit/d6ba1c573134409bca4c3f4d95f608c319ca6c40))
- **assistant:** raise ValueError when guardrails.enabled=false (closes [#1408](https://github.com/NorthlandPositronics/Cogtrix/issues/1408) Vector 3) ([bac3094](https://github.com/NorthlandPositronics/Cogtrix/commit/bac30949adce2129a3395fac5261999fc642d4d9))
- **assistant:** run injection/encoding guardrails in handle_outbound for trusted operators ([201a03d](https://github.com/NorthlandPositronics/Cogtrix/commit/201a03d836704eba243665afcd6231df5ea25334))
- **assistant:** run injection/encoding guardrails in handle_outbound for trusted operators ([6e5186e](https://github.com/NorthlandPositronics/Cogtrix/commit/6e5186eb8cf985ba8403808cdf5cb1c4a7d96bab))
- **assistant:** run injection/encoding guardrails in handle_outbound for trusted operators ([#1406](https://github.com/NorthlandPositronics/Cogtrix/issues/1406)) ([201a03d](https://github.com/NorthlandPositronics/Cogtrix/commit/201a03d836704eba243665afcd6231df5ea25334))
- **assistant:** sanitize campaign goal and instructions before LLM prompts ([#1119](https://github.com/NorthlandPositronics/Cogtrix/issues/1119)) ([2aced22](https://github.com/NorthlandPositronics/Cogtrix/commit/2aced22328afaca521f1124d500691c24401f9d9))
- **assistant:** submit CampaignManager follow-ups to ThreadPoolExecutor ([e9902e1](https://github.com/NorthlandPositronics/Cogtrix/commit/e9902e1b08d4e5a286506d08e514d09b82dc644f))
- **assistant:** submit CampaignManager follow-ups to ThreadPoolExecutor ([d6486f2](https://github.com/NorthlandPositronics/Cogtrix/commit/d6486f20c3b51084dd01f76809a75ad9de630f4d))
- **assistant:** track down upstream None source for InputGuard.check() ([#1455](https://github.com/NorthlandPositronics/Cogtrix/issues/1455)) ([69ca5e3](https://github.com/NorthlandPositronics/Cogtrix/commit/69ca5e360778fce101ab5b0bb7a7b6a44c3a2c12))
- **assistant:** track down upstream None source for InputGuard.check() ([#1455](https://github.com/NorthlandPositronics/Cogtrix/issues/1455)) ([b82bea0](https://github.com/NorthlandPositronics/Cogtrix/commit/b82bea0116825d6a50545c8a9510c0ae8537455f))
- **assistant:** wrap channel.send() in try/except in \_route_response() ([2d206af](https://github.com/NorthlandPositronics/Cogtrix/commit/2d206afc4a8ccd61b16e33c008b8dabe6e7298cd)), closes [#1092](https://github.com/NorthlandPositronics/Cogtrix/issues/1092)
- **assistant:** wrap channel.send() in try/except in \_route_response() ([65b376c](https://github.com/NorthlandPositronics/Cogtrix/commit/65b376c6f4544ede01cd9dea754fe0022cc3c43a)), closes [#1092](https://github.com/NorthlandPositronics/Cogtrix/issues/1092)
- **ci:** add per-test timeout to prevent xdist worker hangs ([ccf1ef5](https://github.com/NorthlandPositronics/Cogtrix/commit/ccf1ef58160725d7a0088dddefac3de7620d253b)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** add trailing \* to test shard glob patterns ([7a77844](https://github.com/NorthlandPositronics/Cogtrix/commit/7a778447b4a8d7f92eeb0dda95c4c5fac0ff535e)), closes [#1387](https://github.com/NorthlandPositronics/Cogtrix/issues/1387)
- **ci:** add trailing \* to test shard glob patterns to fix zero-test-collection ([25a3096](https://github.com/NorthlandPositronics/Cogtrix/commit/25a30968f0c4a4b825d93d3eef033ebffc1b23bc))
- **ci:** auto-pass Gate 2 on release-please PRs ([4bf83a5](https://github.com/NorthlandPositronics/Cogtrix/commit/4bf83a56a0a849cb98b4a0095d4c7f61c3ef7a49))
- **ci:** auto-pass Gate 2 on release-please PRs ([c5a4d71](https://github.com/NorthlandPositronics/Cogtrix/commit/c5a4d711d075e5e4b74a1f90e2500ea8283c7c3e))
- **ci:** bump unit-tests timeout 10→15 min for 4-shard CI runners ([438cfa1](https://github.com/NorthlandPositronics/Cogtrix/commit/438cfa1f933e49c8268b75035b911eb7dc922ecf)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** include .github/workflows/ in source-change filter ([10c4642](https://github.com/NorthlandPositronics/Cogtrix/commit/10c46426bb37bbab034e64eed426f581bc87e0dd)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** inline -m flag to avoid shell quote expansion bug ([3e0af22](https://github.com/NorthlandPositronics/Cogtrix/commit/3e0af22136dc81283ec3450dd86fe31cd96ff6d8)), closes [#1387](https://github.com/NorthlandPositronics/Cogtrix/issues/1387)
- **ci:** install rag extra in CI so FAISS is available during tests ([45a828e](https://github.com/NorthlandPositronics/Cogtrix/commit/45a828e02ff7abb39279bf033a7762a57dea9691)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** make gate2 depend on unit-tests to skip on failure ([6a13609](https://github.com/NorthlandPositronics/Cogtrix/commit/6a136097e63f2f649a597bb045207126827738e7)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** move test_regression_975.py from shard D to shard A ([90a13fb](https://github.com/NorthlandPositronics/Cogtrix/commit/90a13fbb10f062f5b407c3ffa4e8385716583667)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** move test_shell.py and test_http_request.py from shard D to A ([3c14088](https://github.com/NorthlandPositronics/Cogtrix/commit/3c1408816bd4831852c4ae42228e0df5aad5ec9e)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** rebalance unit test shards — move heaviest files from B to A ([d429951](https://github.com/NorthlandPositronics/Cogtrix/commit/d429951c79241c2426a744c6dca82f464704b402)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** restore head -1 in gemma container log dump ([c2e1079](https://github.com/NorthlandPositronics/Cogtrix/commit/c2e10799239b00303170682b7f51a629f5aa90be))
- **ci:** set unit-tests timeout back to 10 minutes ([eb97cad](https://github.com/NorthlandPositronics/Cogtrix/commit/eb97cad930cacec5855df2aafbc2db0176854b54))
- **ci:** split unit tests into 4 balanced shards (A/B/C/D) ([a97e06d](https://github.com/NorthlandPositronics/Cogtrix/commit/a97e06df1294ae6f0a535a9a99b0439c54eaebae)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **ci:** use --dist=loadfile to isolate test files per xdist worker ([2363de5](https://github.com/NorthlandPositronics/Cogtrix/commit/2363de5238f44d7bec663b54c56c2c6bd6bf8972)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **deps:** bump anyio + greenlet in uv.lock to match Dependabot intent ([e00489c](https://github.com/NorthlandPositronics/Cogtrix/commit/e00489ca17bb58d9f2be7e6d5aaec370a4549c04))
- **deps:** bump langchain-classic + langsmith in uv.lock to match ([19b449e](https://github.com/NorthlandPositronics/Cogtrix/commit/19b449ee48b895b0a375a98dc6d15e32eb8a9e1c))
- **deps:** bump orjson 3.11.7 → 3.11.9 in uv.lock to match Dependabot intent ([a1fcde0](https://github.com/NorthlandPositronics/Cogtrix/commit/a1fcde05b2b255ea46269debd282e93dfef7dd91))
- **docs:** remove stale ?token= query param refs from session WebSocket docs ([#1279](https://github.com/NorthlandPositronics/Cogtrix/issues/1279)) ([45951ed](https://github.com/NorthlandPositronics/Cogtrix/commit/45951ed394e859290d92b271ccd123e463644001))
- **docs:** remove stale ?token= query param refs from session WebSocket docs ([#1279](https://github.com/NorthlandPositronics/Cogtrix/issues/1279)) ([6a98458](https://github.com/NorthlandPositronics/Cogtrix/commit/6a984585d161762783a8d3bf8069e9469bd64797))
- **docs:** wrap long line in webui-development-guide.md ([b770809](https://github.com/NorthlandPositronics/Cogtrix/commit/b7708093b787fc06c730f6594f4c74c375487fcc))
- **eval:** classify_invoice stub returns tier field to fix llama3 loop ([3362f2f](https://github.com/NorthlandPositronics/Cogtrix/commit/3362f2f4ad45791b2d39f81af09d6bbb879f3fff))
- **evaluation:** route deepseek provider through factory to enable reasoning_content threading ([df0df81](https://github.com/NorthlandPositronics/Cogtrix/commit/df0df81a35769f99caef13a0347d9daebd00ac4c))
- **evaluation:** route deepseek provider through factory to enable reasoning_content threading ([64db29b](https://github.com/NorthlandPositronics/Cogtrix/commit/64db29ba08ff9f573deb9c28d47424f5d3a5e6c5))
- **gate2:** swap DeepSeek V3 → V4 Flash in smoke matrix to fix flake ([851117b](https://github.com/NorthlandPositronics/Cogtrix/commit/851117b796edd36758917a65e06fd01a1e09a172))
- **mcp:** add **del** to stop event loop on GC for tests without close_all() ([61c4534](https://github.com/NorthlandPositronics/Cogtrix/commit/61c45345e0ac7dbb9425bd9458e13083145ea484)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **mcp:** close call_tool coroutines when \_run() raises before consuming them ([99bb69e](https://github.com/NorthlandPositronics/Cogtrix/commit/99bb69e41dd4ad64b710f6b1777f600d245f062a)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **mcp:** close TOCTOU race in \_run() that orphaned call_tool coroutines ([f55f926](https://github.com/NorthlandPositronics/Cogtrix/commit/f55f926066e4194730804c0589b29a5c42e38fbd)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **mcp:** stop heartbeat task when manager has no connections ([763fe22](https://github.com/NorthlandPositronics/Cogtrix/commit/763fe2277bcf533a7937078ec3581169c5de82cb)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **memory:** fix 3 RLock/thread-safety inconsistencies (closes [#1407](https://github.com/NorthlandPositronics/Cogtrix/issues/1407)) ([adc7cca](https://github.com/NorthlandPositronics/Cogtrix/commit/adc7cca055d5b6c96da7077e1385feaca29d120c))
- **memory:** fix 3 RLock/thread-safety inconsistencies in memory subsystem ([4a1c494](https://github.com/NorthlandPositronics/Cogtrix/commit/4a1c494e732a81387535f8abb3a805af0fb3d45f))
- **memory:** fix AB/BA deadlock in ReasoningMemoryManager.clear() ([#1402](https://github.com/NorthlandPositronics/Cogtrix/issues/1402)) ([0757e21](https://github.com/NorthlandPositronics/Cogtrix/commit/0757e214967965fad2cc9960ce9d5d1d57a10e5d))
- **memory:** fix AB/BA deadlock in ReasoningMemoryManager.clear() ([#1402](https://github.com/NorthlandPositronics/Cogtrix/issues/1402)) ([b9e8485](https://github.com/NorthlandPositronics/Cogtrix/commit/b9e84855b83da5ca95fd1701131a716166fc5fe8))
- **memory:** fix AB/BA deadlock in ReasoningMemoryManager.clear() ([#1402](https://github.com/NorthlandPositronics/Cogtrix/issues/1402)) ([#1403](https://github.com/NorthlandPositronics/Cogtrix/issues/1403)) ([0757e21](https://github.com/NorthlandPositronics/Cogtrix/commit/0757e214967965fad2cc9960ce9d5d1d57a10e5d))
- **memory:** handle list-type LLM response content in summarizer ([4dd580f](https://github.com/NorthlandPositronics/Cogtrix/commit/4dd580ffc1fd2918d30e0222b601b184f328313a))
- **memory:** handle list-type LLM response content in summarizer ([9e06702](https://github.com/NorthlandPositronics/Cogtrix/commit/9e06702a91786945946481348d57212cee9933f5))
- **memory:** preserve None guard in summarizer coercion ([9ebcc78](https://github.com/NorthlandPositronics/Cogtrix/commit/9ebcc7858faa1e4a779767e4ecc748570366d7d1))
- **memory:** protect \_pending_user_ts with \_hybrid_lock (closes [#1344](https://github.com/NorthlandPositronics/Cogtrix/issues/1344)) ([000af47](https://github.com/NorthlandPositronics/Cogtrix/commit/000af47a601be6406dd1ed93562014c4297490f6))
- **memory:** protect \_pending_user_ts with \_hybrid_lock (closes [#1344](https://github.com/NorthlandPositronics/Cogtrix/issues/1344)) ([2ea129c](https://github.com/NorthlandPositronics/Cogtrix/commit/2ea129cda4c650f32b89e3c6f4466e40338c8c3a))
- **memory:** protect \_tokens_since_summary increments with \_hybrid_lock ([#1405](https://github.com/NorthlandPositronics/Cogtrix/issues/1405)) ([699c849](https://github.com/NorthlandPositronics/Cogtrix/commit/699c8496ba87b80a4ffeee66de968862e5eb1ec5))
- **memory:** protect \_tokens_since_summary increments with \_hybrid_lock (closes [#1295](https://github.com/NorthlandPositronics/Cogtrix/issues/1295)) ([699c849](https://github.com/NorthlandPositronics/Cogtrix/commit/699c8496ba87b80a4ffeee66de968862e5eb1ec5))
- **memory:** protect \_tokens_since_summary increments with \_hybrid_lock (closes [#1295](https://github.com/NorthlandPositronics/Cogtrix/issues/1295)) ([3f9e8c2](https://github.com/NorthlandPositronics/Cogtrix/commit/3f9e8c2067095bf63e346ec2b3b026cc4920f110))
- **memory:** replace ThreadPoolExecutor context manager with manual shutdown in compress_to_tier() ([6c188b7](https://github.com/NorthlandPositronics/Cogtrix/commit/6c188b7912efae6a6c32a9eb44cb0005d690a808))
- **memory:** replace ThreadPoolExecutor context manager with manual shutdown in compress_to_tier() ([25a47c2](https://github.com/NorthlandPositronics/Cogtrix/commit/25a47c23bf2be71e3efe880314e2bfa97bae429e)), closes [#1155](https://github.com/NorthlandPositronics/Cogtrix/issues/1155)
- **memory:** revert \_hybrid_lock to Lock() per [#1401](https://github.com/NorthlandPositronics/Cogtrix/issues/1401) split-method pattern ([318ea1d](https://github.com/NorthlandPositronics/Cogtrix/commit/318ea1d37d74b83a0395c7885fcee16aa6c3743b))
- **memory:** snapshot self.\_messages under \_mode_lock before scheduling (closes [#1342](https://github.com/NorthlandPositronics/Cogtrix/issues/1342), [#1296](https://github.com/NorthlandPositronics/Cogtrix/issues/1296)) ([a5ed72a](https://github.com/NorthlandPositronics/Cogtrix/commit/a5ed72a502a72e084b6bbd000d64d1f42eb27994))
- **memory:** snapshot self.\_messages under \_mode_lock before scheduling (closes [#1342](https://github.com/NorthlandPositronics/Cogtrix/issues/1342), [#1296](https://github.com/NorthlandPositronics/Cogtrix/issues/1296)) ([f8a9003](https://github.com/NorthlandPositronics/Cogtrix/commit/f8a9003e7d95b9ce0efd3e6a4663bb456e5234c7))
- **orchestration:** add classify to action-intent tool verb regex ([7d493cb](https://github.com/NorthlandPositronics/Cogtrix/commit/7d493cb8f31012284f86d798a823f80c7523f993)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **orchestration:** address PR review feedback on TOCTOU race fix ([e450bdc](https://github.com/NorthlandPositronics/Cogtrix/commit/e450bdcd34f18757ff7e15ad4f178057cb216c33))
- **orchestration:** detect past-tense hallucinated tool completions ([a71b7dd](https://github.com/NorthlandPositronics/Cogtrix/commit/a71b7dd30b98799feef31b6cf240bd5d2ed40787))
- **orchestration:** eliminate TOCTOU race in parallel tool duplicate detection (closes [#1293](https://github.com/NorthlandPositronics/Cogtrix/issues/1293)) ([307a69d](https://github.com/NorthlandPositronics/Cogtrix/commit/307a69d53aa2ac0f943b3ac820fbc1a6825bd7bd))
- **orchestration:** eliminate TOCTOU race in parallel tool duplicate detection (closes [#1293](https://github.com/NorthlandPositronics/Cogtrix/issues/1293)) ([e686277](https://github.com/NorthlandPositronics/Cogtrix/commit/e686277580ffbeacc4288930d8f74513e071cc61))
- **orchestration:** rewrite thinking-break prompt to drop scenario-forbidden verbiage ([4b98439](https://github.com/NorthlandPositronics/Cogtrix/commit/4b98439085df4ef7d484f379799b872908f30940))
- **orchestration:** structurally reset per-run state via PerRunState dataclass ([8aa665e](https://github.com/NorthlandPositronics/Cogtrix/commit/8aa665e94a4a2389e1975389d35de44b67416741))
- **orchestration:** structurally reset per-run state via PerRunState dataclass ([#1410](https://github.com/NorthlandPositronics/Cogtrix/issues/1410)) ([a5c8c28](https://github.com/NorthlandPositronics/Cogtrix/commit/a5c8c285093502f35f5cb1b42afebe53d01157c8))
- **orchestration:** structurally reset PerRunState via fresh instance copy ([#1292](https://github.com/NorthlandPositronics/Cogtrix/issues/1292)) ([275fec6](https://github.com/NorthlandPositronics/Cogtrix/commit/275fec64f7ebc88f8daea85c265614825e5edb5b))
- **orchestration:** structurally reset PerRunState via fresh instance copy ([#1292](https://github.com/NorthlandPositronics/Cogtrix/issues/1292)) ([17dd908](https://github.com/NorthlandPositronics/Cogtrix/commit/17dd9083843315063932e9e915f2cee722bab597))
- **orchestration:** tighten thinking-break to single forced path ([4c8ad27](https://github.com/NorthlandPositronics/Cogtrix/commit/4c8ad27f271c485e5bdd18b23d4d4a512ed373eb))
- **security:** explicit UUID v4 validation in cross_workspace (CWE-22, closes [#1368](https://github.com/NorthlandPositronics/Cogtrix/issues/1368)) ([5b61c59](https://github.com/NorthlandPositronics/Cogtrix/commit/5b61c598fadc46bdf56b60e28605c778b6de072e))
- **security:** explicit UUID v4 validation in cross_workspace (CWE-22, closes [#1368](https://github.com/NorthlandPositronics/Cogtrix/issues/1368)) ([2e4fdac](https://github.com/NorthlandPositronics/Cogtrix/commit/2e4fdac457a68f433d05aa65043ed27be278dfc8))
- **tests/memory:** apply Black formatting to test_code.py ([18abc5c](https://github.com/NorthlandPositronics/Cogtrix/commit/18abc5cd28092dcb9ecdef4bba5b744246b0ae81))
- **tests:** add content=None regression test for summarizer ([5c0ede3](https://github.com/NorthlandPositronics/Cogtrix/commit/5c0ede3c7b2fd63b68c04c08205a0194f0b13017))
- **tests:** force multiprocessing spawn start method to avoid fork-in-threaded-process ([c0baeee](https://github.com/NorthlandPositronics/Cogtrix/commit/c0baeeecb7f7698da5425374efde11b9c5166b1f)), closes [#1390](https://github.com/NorthlandPositronics/Cogtrix/issues/1390)
- **tests:** invert flaky test assertion + add xfail marker for [#1431](https://github.com/NorthlandPositronics/Cogtrix/issues/1431) ([ec66106](https://github.com/NorthlandPositronics/Cogtrix/commit/ec661068432768c4e6818febf875fa284b03de76))
- **tests:** invert flaky test assertion + add xfail marker for [#1431](https://github.com/NorthlandPositronics/Cogtrix/issues/1431) ([98c8a8d](https://github.com/NorthlandPositronics/Cogtrix/commit/98c8a8dc834f199b6f004fa91d7300238eae7a48))
- **tests:** patch save_faiss_store in test_metadata_passed_to_document ([117e077](https://github.com/NorthlandPositronics/Cogtrix/commit/117e0771022dbf5b60a048d6cffbeeb24a4b98a1))
- **tests:** remove unused module-level threading import (ruff F401/F811) ([7359c85](https://github.com/NorthlandPositronics/Cogtrix/commit/7359c851ba1e27e880f96f085a385742d9544eef))
- **tests:** rename duplicate test_skip_trusted_checks_bypasses_rate_limit ([1d69d17](https://github.com/NorthlandPositronics/Cogtrix/commit/1d69d17adef8eb4c735d445d2a7ba3d1e6a1d3ab))
- **tests:** resolve 4 failing unit shards in PR [#1432](https://github.com/NorthlandPositronics/Cogtrix/issues/1432) ([8a2a284](https://github.com/NorthlandPositronics/Cogtrix/commit/8a2a284893a60405ce5c74301dbc09bb1f0942d3))
- **tests:** update handle_outbound guardrail test for new trusted-operator contract ([a638aec](https://github.com/NorthlandPositronics/Cogtrix/commit/a638aeceb69194ed5ac301db35438d5f5f49d25b))
- **tests:** update serpapi exception test for sanitized error messages ([c25eee8](https://github.com/NorthlandPositronics/Cogtrix/commit/c25eee84b7a8a9982fb0596c4fb37a1d65f081e1)), closes [#1424](https://github.com/NorthlandPositronics/Cogtrix/issues/1424)
- **tests:** use separate thread with Barrier to test nonblocking lock behavior ([98720ae](https://github.com/NorthlandPositronics/Cogtrix/commit/98720aede2a4ad473abc2b7d3c127c2b058e4c2f))
- **tools/python_exec:** fix \_AVAILABLE_OPTIONAL not updated when module already in SAFE_MODULES ([0035fcd](https://github.com/NorthlandPositronics/Cogtrix/commit/0035fcdba565db85367640b7a0427b9067a39579))
- **tools/python_exec:** fix \_AVAILABLE_OPTIONAL not updated when module already in SAFE_MODULES ([78d2505](https://github.com/NorthlandPositronics/Cogtrix/commit/78d250536f05d507f938ff26c2d2529a2878034a)), closes [#1291](https://github.com/NorthlandPositronics/Cogtrix/issues/1291)
- **tools/shell:** block process substitution &lt;() &gt;() and sanitize subprocess env (closes [#1238](https://github.com/NorthlandPositronics/Cogtrix/issues/1238), [#1239](https://github.com/NorthlandPositronics/Cogtrix/issues/1239)) ([511098d](https://github.com/NorthlandPositronics/Cogtrix/commit/511098de49b2682bd7a6ef85db21ad1752a07ea3))
- **tools/shell:** block process substitution and sanitize subprocess env ([#1413](https://github.com/NorthlandPositronics/Cogtrix/issues/1413)) ([1d80c69](https://github.com/NorthlandPositronics/Cogtrix/commit/1d80c69aff8bfb10d2cc2a6e9f225d67bc6cbd10))
- **tools/shell:** block process substitution and sanitize subprocess env (closes [#1238](https://github.com/NorthlandPositronics/Cogtrix/issues/1238), [#1239](https://github.com/NorthlandPositronics/Cogtrix/issues/1239)) ([1d80c69](https://github.com/NorthlandPositronics/Cogtrix/commit/1d80c69aff8bfb10d2cc2a6e9f225d67bc6cbd10))
- **tools:** add file size check to patch_file before read_text() ([#1242](https://github.com/NorthlandPositronics/Cogtrix/issues/1242)) ([4a2b642](https://github.com/NorthlandPositronics/Cogtrix/commit/4a2b6428fde6d1ff7e913b260c4391e39474b3d5))
- **tools:** add file size check to patch_file before read_text() ([#1418](https://github.com/NorthlandPositronics/Cogtrix/issues/1418)) ([5c4c1f7](https://github.com/NorthlandPositronics/Cogtrix/commit/5c4c1f706a01d9a1b8df9cbe7595bd2f5ba7dfce))
- **tools:** add ip.is_multicast to SSRF block in http_request.py (closes [#1416](https://github.com/NorthlandPositronics/Cogtrix/issues/1416)) ([a7085e1](https://github.com/NorthlandPositronics/Cogtrix/commit/a7085e1a39524999e2d401c6c8fa0c32a4c2dd3e))
- **tools:** add ip.is_multicast to SSRF block in http_request.py (closes [#1416](https://github.com/NorthlandPositronics/Cogtrix/issues/1416)) ([7411368](https://github.com/NorthlandPositronics/Cogtrix/commit/74113688e907b1f00aa1798cfcfa3d2c6c39067a))
- **tools:** address Owen TL review findings on PR [#1432](https://github.com/NorthlandPositronics/Cogtrix/issues/1432) ([05358f0](https://github.com/NorthlandPositronics/Cogtrix/commit/05358f0f22420c66494229a112806b3d2214b5ee)), closes [#1424](https://github.com/NorthlandPositronics/Cogtrix/issues/1424)
- **tools:** cap shell output before memory exhaustion (closes [#1241](https://github.com/NorthlandPositronics/Cogtrix/issues/1241)) ([05697c3](https://github.com/NorthlandPositronics/Cogtrix/commit/05697c3ce2a4713168fe03bd82c6790062ee13bf))
- **tools:** cap shell output before memory exhaustion (closes [#1241](https://github.com/NorthlandPositronics/Cogtrix/issues/1241)) ([d39b8eb](https://github.com/NorthlandPositronics/Cogtrix/commit/d39b8eb7eea581d2a71ebc03f0079defedaa5036))
- **tools:** complete [#1259](https://github.com/NorthlandPositronics/Cogtrix/issues/1259) — export configure_datascience_modules, add regression tests, fix spawn propagation ([a38cd9d](https://github.com/NorthlandPositronics/Cogtrix/commit/a38cd9dd7879f9b40dc6e211235b8168329900e5))
- **tools:** expose configure_datascience_modules in **all** and add regression tests ([1d71ac4](https://github.com/NorthlandPositronics/Cogtrix/commit/1d71ac4cf629bebf48465a194d8e6f0c4fdc8a5c)), closes [#1259](https://github.com/NorthlandPositronics/Cogtrix/issues/1259)
- **tools:** log warning when loop limiter AST transformation fails (closes [#1203](https://github.com/NorthlandPositronics/Cogtrix/issues/1203)) ([eefb98c](https://github.com/NorthlandPositronics/Cogtrix/commit/eefb98c6c39661db4e7368590017aa98a6f58d8d))
- **tools:** log warning when loop limiter AST transformation fails (closes [#1203](https://github.com/NorthlandPositronics/Cogtrix/issues/1203)) ([ad68113](https://github.com/NorthlandPositronics/Cogtrix/commit/ad681139020351e5aaa237924c7988775e9b4338))
- **tools:** refresh TOOL_CONFIG description after configure_datascience_modules (closes [#1291](https://github.com/NorthlandPositronics/Cogtrix/issues/1291)) ([4351978](https://github.com/NorthlandPositronics/Cogtrix/commit/4351978b89230d00629c808666307b6190ae111d))
- **tools:** refresh TOOL_CONFIG description after configure_datascience_modules (closes [#1291](https://github.com/NorthlandPositronics/Cogtrix/issues/1291)) ([228e2b3](https://github.com/NorthlandPositronics/Cogtrix/commit/228e2b36152e3275ee1e188cde37e4dca08a3126))
- **tools:** sanitize exception messages across 10+ tool files ([d18127d](https://github.com/NorthlandPositronics/Cogtrix/commit/d18127d561ad68350e229e726140243987405637))
- **tools:** sanitize exception messages across 10+ tool files ([871b869](https://github.com/NorthlandPositronics/Cogtrix/commit/871b869d0d47966b00e0691b5e57df34956e54fc)), closes [#1424](https://github.com/NorthlandPositronics/Cogtrix/issues/1424)
- **tools:** sanitize exception messages across 10+ tool files ([#1424](https://github.com/NorthlandPositronics/Cogtrix/issues/1424)) ([d18127d](https://github.com/NorthlandPositronics/Cogtrix/commit/d18127d561ad68350e229e726140243987405637))
- **tools:** sanitize exception messages in http_get/http_post ([88658a8](https://github.com/NorthlandPositronics/Cogtrix/commit/88658a899fbebbfe4890eda2b0ce15ed17abf8fd))
- **tools:** sanitize exception messages in http_get/http_post ([#1421](https://github.com/NorthlandPositronics/Cogtrix/issues/1421)) ([e576ea3](https://github.com/NorthlandPositronics/Cogtrix/commit/e576ea31330ccad58d3e10fef78d4c3445111e03))
- **tools:** sanitize exception messages in http_get/http_post (closes [#1420](https://github.com/NorthlandPositronics/Cogtrix/issues/1420)) ([e576ea3](https://github.com/NorthlandPositronics/Cogtrix/commit/e576ea31330ccad58d3e10fef78d4c3445111e03))

### Performance Improvements

- **tests:** patch hung-pool timeouts down via module constants ([9099af4](https://github.com/NorthlandPositronics/Cogtrix/commit/9099af43916197f2b90e8c95fd310bfe37375f8c))

### Documentation

- **memory:** add lock hierarchy documentation to BaseMemoryManager ([b64441f](https://github.com/NorthlandPositronics/Cogtrix/commit/b64441f433c0a0a3ac7f9e49bff739ec8ca53d1a))
- **memory:** add lock hierarchy documentation to BaseMemoryManager ([d632e27](https://github.com/NorthlandPositronics/Cogtrix/commit/d632e276487338309aa0be10df8d986c18a9904b))
- **memory:** add lock hierarchy documentation to BaseMemoryManager ([#1419](https://github.com/NorthlandPositronics/Cogtrix/issues/1419)) ([b64441f](https://github.com/NorthlandPositronics/Cogtrix/commit/b64441f433c0a0a3ac7f9e49bff739ec8ca53d1a))

## [0.2.7](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.6...v0.2.7) (2026-05-13)

### Features

- [#480](https://github.com/NorthlandPositronics/Cogtrix/issues/480) Step 7 CI wiring — Gate 2 runs on PRs to release/\* branches ([#571](https://github.com/NorthlandPositronics/Cogtrix/issues/571)) ([7ea28fc](https://github.com/NorthlandPositronics/Cogtrix/commit/7ea28fce72c2c3a7c00915ea107fde0aa2c51505))
- **#380:** token-based summary TTL (Layer 1a) ([31531db](https://github.com/NorthlandPositronics/Cogtrix/commit/31531db3a89e13b50dd9fe4f4490a393c8839489))
- **#380:** token-based summary TTL (Layer 1a) ([60a9e34](https://github.com/NorthlandPositronics/Cogtrix/commit/60a9e3413f24ce70ba593f3ce1bd394468067b9f)), closes [#380](https://github.com/NorthlandPositronics/Cogtrix/issues/380)
- add admin system stats endpoint (issue [#617](https://github.com/NorthlandPositronics/Cogtrix/issues/617)) ([3ef6779](https://github.com/NorthlandPositronics/Cogtrix/commit/3ef67799372182fe925dcf7d690f8a02406da3a6))
- add OpenTelemetry tracing ([#422](https://github.com/NorthlandPositronics/Cogtrix/issues/422)) ([6b923fb](https://github.com/NorthlandPositronics/Cogtrix/commit/6b923fb3f53b9da8fa835aef321f05b48c3f5383))
- add Prometheus alert rules for Cogtrix monitoring ([#645](https://github.com/NorthlandPositronics/Cogtrix/issues/645)) ([8619c0b](https://github.com/NorthlandPositronics/Cogtrix/commit/8619c0b420ddb93c65a10387466fe98fcc386d84)), closes [#610](https://github.com/NorthlandPositronics/Cogtrix/issues/610)
- add quality dashboard report ([#573](https://github.com/NorthlandPositronics/Cogtrix/issues/573)) ([8b5ebad](https://github.com/NorthlandPositronics/Cogtrix/commit/8b5ebade9690e80ddc3d2da700547c32fde13ffe))
- add superadmin-only admin org list with created_after filter ([#615](https://github.com/NorthlandPositronics/Cogtrix/issues/615)) ([#700](https://github.com/NorthlandPositronics/Cogtrix/issues/700)) ([5ad5ae2](https://github.com/NorthlandPositronics/Cogtrix/commit/5ad5ae2626056ae86df7c8bcae84a170b459f910))
- add trace_id and span_id to structured JSON logs (ISSUE [#609](https://github.com/NorthlandPositronics/Cogtrix/issues/609)) ([#636](https://github.com/NorthlandPositronics/Cogtrix/issues/636)) ([8473c2c](https://github.com/NorthlandPositronics/Cogtrix/commit/8473c2ce150edd2ade3251bed913372707441627))
- **admin:** Phase 2.6.1 — admin org list API with cursor pagination and filters ([f2ef130](https://github.com/NorthlandPositronics/Cogtrix/commit/f2ef130d0365b9ee78113b96824293ef31d49f5e))
- **admin:** Phase 2.6.1 — admin org list API with cursor pagination and filters ([2e4498e](https://github.com/NorthlandPositronics/Cogtrix/commit/2e4498eff32ee807a4b646429a4593f23785538f))
- **admin:** Phase 2.6.2 — org-level usage and audit endpoints ([37ff241](https://github.com/NorthlandPositronics/Cogtrix/commit/37ff241778d7aa78baec7cf7a8aefd65460b9d3f))
- **admin:** Phase 2.6.2 — org-level usage and audit endpoints ([5eb2ad1](https://github.com/NorthlandPositronics/Cogtrix/commit/5eb2ad182d4ef0dbd7f200eac7e526f061a63f40))
- **admin:** Phase 2.6.3 — global system stats endpoint ([d91e008](https://github.com/NorthlandPositronics/Cogtrix/commit/d91e00866ca4f60fe7e0fc6c551cec04ff0dd4ee))
- **admin:** Phase 2.6.3 — global system stats endpoint ([d9228d5](https://github.com/NorthlandPositronics/Cogtrix/commit/d9228d54f100d461384c162ad37ae369c67b7e56))
- **analysis:** add session_metrics.py — automated behavioral metrics from logs ([#483](https://github.com/NorthlandPositronics/Cogtrix/issues/483)) ([#507](https://github.com/NorthlandPositronics/Cogtrix/issues/507)) ([1060bfe](https://github.com/NorthlandPositronics/Cogtrix/commit/1060bfe168c2587a5f74fa733fcf8ef7f34fe28f))
- **ci:** run Gate 2 LLM smoke eval on PRs to next, main, and release/\* (closes [#1059](https://github.com/NorthlandPositronics/Cogtrix/issues/1059)) ([#1063](https://github.com/NorthlandPositronics/Cogtrix/issues/1063)) ([a6ee350](https://github.com/NorthlandPositronics/Cogtrix/commit/a6ee3508fb7c39a79748a706727dae51356f1c88))
- cron task context inheritance ([5cc13e1](https://github.com/NorthlandPositronics/Cogtrix/commit/5cc13e16bd6d09682431abf2a55eed19145a82ab))
- **cron:** add inherited session context ([6d27b97](https://github.com/NorthlandPositronics/Cogtrix/commit/6d27b97e96ba1e91e75e69bfb21077f8f2fcfa57))
- embed short git commit hash in version string (closes [#871](https://github.com/NorthlandPositronics/Cogtrix/issues/871)) ([#893](https://github.com/NorthlandPositronics/Cogtrix/issues/893)) ([29879d7](https://github.com/NorthlandPositronics/Cogtrix/commit/29879d7d99125b8e7d8fe95a1c1634aaf6ac3272))
- **eval:** add 5 Finance/Procurement Gate 2 scenario YAMLs ([#481](https://github.com/NorthlandPositronics/Cogtrix/issues/481)) ([#508](https://github.com/NorthlandPositronics/Cogtrix/issues/508)) ([b6d0a21](https://github.com/NorthlandPositronics/Cogtrix/commit/b6d0a2169df49f53cc5995649644bb542d843281))
- **eval:** add three Gate 2 scenario YAMLs ([#566](https://github.com/NorthlandPositronics/Cogtrix/issues/566)) ([a48d35d](https://github.com/NorthlandPositronics/Cogtrix/commit/a48d35de253c74a21edebfecfe1e22bbf11224cc))
- **eval:** dashboard.py + unit tests for Gate 2 results visualization ([#484](https://github.com/NorthlandPositronics/Cogtrix/issues/484)) ([#512](https://github.com/NorthlandPositronics/Cogtrix/issues/512)) ([dd0cd11](https://github.com/NorthlandPositronics/Cogtrix/commit/dd0cd1152c0e739fdc1b170f00ff285cd03d6c49))
- **eval:** Gate 2 API key priority with fallback (OPENROUTER → CEREBRAS → DEEPSEEK → OPENAI → ANTHROPIC) ([#1066](https://github.com/NorthlandPositronics/Cogtrix/issues/1066)) ([ac4d875](https://github.com/NorthlandPositronics/Cogtrix/commit/ac4d875e4b1eda798007c9a61f58f91847084307))
- **eval:** Gate 2 evaluation infrastructure — multi-model domain scenario runner ([4636d8a](https://github.com/NorthlandPositronics/Cogtrix/commit/4636d8adc689e56d861c505d92e19dd51d83056e))
- **eval:** Gate 2 evaluation infrastructure — multi-model domain scenario runner ([2ac290a](https://github.com/NorthlandPositronics/Cogtrix/commit/2ac290a3f4db4d25addc809e05ff1521fd0b5de9))
- **eval:** judge.py — LLM-as-judge scoring for Gate 2 scenarios ([#482](https://github.com/NorthlandPositronics/Cogtrix/issues/482)) ([#514](https://github.com/NorthlandPositronics/Cogtrix/issues/514)) ([a082e26](https://github.com/NorthlandPositronics/Cogtrix/commit/a082e2687de708d0667e46f945ab38ed3958043f))
- **eval:** typed stub-tool registry restores Gate 2 to 5-model 35/35 ([cd7d648](https://github.com/NorthlandPositronics/Cogtrix/commit/cd7d648ede46352787107095feef585fde1f2ee5))
- **eval:** typed stub-tool registry restores Gate 2 to 5-model 35/35 ([4f56ce4](https://github.com/NorthlandPositronics/Cogtrix/commit/4f56ce4df3b32d4c2efa39b1013fcb72c4e13324))
- extract SlashCommandRegistry to src/cli/commands.py ([#572](https://github.com/NorthlandPositronics/Cogtrix/issues/572)) ([8213f3c](https://github.com/NorthlandPositronics/Cogtrix/commit/8213f3cb6102914ecc6f57f8206b6a90bffb27db))
- **gate2:** add 4 regression/safety scenarios and cost-ceiling check ([#36](https://github.com/NorthlandPositronics/Cogtrix/issues/36)) ([#1267](https://github.com/NorthlandPositronics/Cogtrix/issues/1267)) ([889c59e](https://github.com/NorthlandPositronics/Cogtrix/commit/889c59e755c389c24cf1982283ea05fd57400a0b))
- graceful shutdown — SIGTERM handler, WebSocket drain, DB close (issue [#603](https://github.com/NorthlandPositronics/Cogtrix/issues/603)) ([#653](https://github.com/NorthlandPositronics/Cogtrix/issues/653)) ([7de07eb](https://github.com/NorthlandPositronics/Cogtrix/commit/7de07ebb4d690388f5659858b754c0b80d10ee77))
- implement OpenTelemetry 5-layer tracing (ISSUE [#606](https://github.com/NorthlandPositronics/Cogtrix/issues/606)) ([#638](https://github.com/NorthlandPositronics/Cogtrix/issues/638)) ([1873d14](https://github.com/NorthlandPositronics/Cogtrix/commit/1873d144ae9ff6076c21eccaaf787e0820b558b4))
- implement Prometheus metrics endpoint with 7 observability metrics ([#607](https://github.com/NorthlandPositronics/Cogtrix/issues/607)) ([#629](https://github.com/NorthlandPositronics/Cogtrix/issues/629)) ([ce09651](https://github.com/NorthlandPositronics/Cogtrix/commit/ce096515d6fac2aea1f2b1fcf90265107b45c597))
- **ldap:** connection pooling for LDAP/AD sync + group search (Phase 2.1.3) ([8ec9064](https://github.com/NorthlandPositronics/Cogtrix/commit/8ec9064334dee46d002d778e261c98a0d3bf427e))
- **ldap:** Phase 2.1.4 — map LDAP/AD group memberships to Cogtrix roles on sync ([774e38c](https://github.com/NorthlandPositronics/Cogtrix/commit/774e38c3c4db249e2a161e96acd9abf15c95d76d))
- **ldap:** Phase 2.1.4 — map LDAP/AD group memberships to Cogtrix roles on sync ([dad0367](https://github.com/NorthlandPositronics/Cogtrix/commit/dad036707ac771736caf69a994c579ad47a37e9e)), closes [#398](https://github.com/NorthlandPositronics/Cogtrix/issues/398)
- **logging:** add structured JSON logs ([47d6e2a](https://github.com/NorthlandPositronics/Cogtrix/commit/47d6e2a294e768798e80428aea9c1b7e74f252c7))
- **logging:** structured JSON logs with session/request context ([#415](https://github.com/NorthlandPositronics/Cogtrix/issues/415)) ([8860b42](https://github.com/NorthlandPositronics/Cogtrix/commit/8860b420584e0bfa7c19d2fa7ed46c4c1b949aeb))
- **memory:** deterministic recall test harness with synthetic corpus ([#133](https://github.com/NorthlandPositronics/Cogtrix/issues/133)) ([29b2c76](https://github.com/NorthlandPositronics/Cogtrix/commit/29b2c761166213ea991f6b42b8c9ac3218d3766b))
- **memory:** deterministic recall test harness with synthetic corpus ([#133](https://github.com/NorthlandPositronics/Cogtrix/issues/133)) ([b6dd00e](https://github.com/NorthlandPositronics/Cogtrix/commit/b6dd00e67d941e1391ebcffcce127cff0868c84e))
- **metrics:** Phase 2.4.2 — Prometheus metrics endpoint ([a2e8a37](https://github.com/NorthlandPositronics/Cogtrix/commit/a2e8a3738b973d2df25e9b6300bcb638ba765d49))
- **metrics:** Phase 2.4.2 — Prometheus metrics endpoint ([bc3fc5e](https://github.com/NorthlandPositronics/Cogtrix/commit/bc3fc5eeecc46d4eb0449a4fec035916499d29b1))
- **metrics:** wire session metrics write after CLI run_agent() ([#480](https://github.com/NorthlandPositronics/Cogtrix/issues/480) Step 8) ([#537](https://github.com/NorthlandPositronics/Cogtrix/issues/537)) ([34bed53](https://github.com/NorthlandPositronics/Cogtrix/commit/34bed53e587e04e112cac956bea9a32884bec178))
- **metrics:** wire tool_calls_total and llm_tokens_total into orchestration ([8b3ebcd](https://github.com/NorthlandPositronics/Cogtrix/commit/8b3ebcdbb578a5cd35647f9c3d8dba9a02eb45cc)), closes [#420](https://github.com/NorthlandPositronics/Cogtrix/issues/420)
- **metrics:** wire tool_calls_total and llm_tokens_total into orchestration ([#420](https://github.com/NorthlandPositronics/Cogtrix/issues/420)) ([f11e493](https://github.com/NorthlandPositronics/Cogtrix/commit/f11e4938add72f8a511cd44b85d435b7ac6f7351))
- **observability:** OpenTelemetry HTTP + LLM + tool tracing ([#422](https://github.com/NorthlandPositronics/Cogtrix/issues/422)) ([bc8e82f](https://github.com/NorthlandPositronics/Cogtrix/commit/bc8e82f9229d6fd815f4030a2e5fac61ab7faa20))
- permission model and role-permission matrix (issue [#594](https://github.com/NorthlandPositronics/Cogtrix/issues/594)) ([#658](https://github.com/NorthlandPositronics/Cogtrix/issues/658)) ([4c54a54](https://github.com/NorthlandPositronics/Cogtrix/commit/4c54a54ec2465ba91a07d4094065faeecc0b24bc))
- quality ratchet — load_baseline, check_ratchet, update_baseline ([#570](https://github.com/NorthlandPositronics/Cogtrix/issues/570)) ([25f6500](https://github.com/NorthlandPositronics/Cogtrix/commit/25f650056b0d7ad18dfbccaeb1a731b991aca020))
- **quality:** add Gate 2 LLM-as-judge scoring ([#567](https://github.com/NorthlandPositronics/Cogtrix/issues/567)) ([7ecfdf9](https://github.com/NorthlandPositronics/Cogtrix/commit/7ecfdf94a87277af4a3b1267dde5f7627e963f40))
- **quality:** log Gate 2 judge scores in CI ([#568](https://github.com/NorthlandPositronics/Cogtrix/issues/568)) ([a0810d4](https://github.com/NorthlandPositronics/Cogtrix/commit/a0810d46e7c87903916a44dbc0b60c6d436fe545))
- RBAC enforcement middleware with require() dependency (issue [#596](https://github.com/NorthlandPositronics/Cogtrix/issues/596)) ([#668](https://github.com/NorthlandPositronics/Cogtrix/issues/668)) ([c31f080](https://github.com/NorthlandPositronics/Cogtrix/commit/c31f080dd4edc26ca0ab0991f36af2e19c12d8d1))
- role assignment API with SAML mapping and audit logging (issue [#595](https://github.com/NorthlandPositronics/Cogtrix/issues/595)) ([#664](https://github.com/NorthlandPositronics/Cogtrix/issues/664)) ([21cd2b8](https://github.com/NorthlandPositronics/Cogtrix/commit/21cd2b892ad225f094d3aaa82d7faec909414482))
- superadmin impersonation with audit logging (issue [#618](https://github.com/NorthlandPositronics/Cogtrix/issues/618)) ([01acc11](https://github.com/NorthlandPositronics/Cogtrix/commit/01acc1132eb41f90fb3f7085cdc8a4a7579efa50))
- **tools:** add cogtrix_slack_post_message wrapper with markdown-to-mrkdwn conversion ([#333](https://github.com/NorthlandPositronics/Cogtrix/issues/333)) ([134964a](https://github.com/NorthlandPositronics/Cogtrix/commit/134964a6467d3f64b9f5037c0e816d703fc2e3c2))
- **tools:** add cogtrix_slack_post_message wrapper with markdown-to-mrkdwn conversion ([#333](https://github.com/NorthlandPositronics/Cogtrix/issues/333)) ([4ef96e5](https://github.com/NorthlandPositronics/Cogtrix/commit/4ef96e5280160454c6668095717d05a7d2a2309b))

### Bug Fixes

- \_compute_file_diff logs errors and raises DiffComputationError (closes [#812](https://github.com/NorthlandPositronics/Cogtrix/issues/812)) ([#877](https://github.com/NorthlandPositronics/Cogtrix/issues/877)) ([39ba29d](https://github.com/NorthlandPositronics/Cogtrix/commit/39ba29de7abba2c9e7c4f3cf0ec046a7d0ec311e))
- \_msg_tokens in memory/manager.py and tier_cache.py handles dict content blocks (issue [#793](https://github.com/NorthlandPositronics/Cogtrix/issues/793)) ([087b9d1](https://github.com/NorthlandPositronics/Cogtrix/commit/087b9d16b05c148c3022cf3839bb7d96c8a83694))
- **#476:** add Slack duplicate-status cooldown ([3d5b119](https://github.com/NorthlandPositronics/Cogtrix/commit/3d5b1196a17d7cbb9899f6ef92a6d6df491a22b1))
- 911: bounded directory traversal for [@path](https://github.com/path) expansion (traversal-only) ([#941](https://github.com/NorthlandPositronics/Cogtrix/issues/941)) ([8550f39](https://github.com/NorthlandPositronics/Cogtrix/commit/8550f39cb7225d4ea4f81453a815ce98ccc229ca))
- add API key and OIDC auth to WebSocket endpoint (issue [#665](https://github.com/NorthlandPositronics/Cogtrix/issues/665)) ([#687](https://github.com/NorthlandPositronics/Cogtrix/issues/687)) ([bd21af7](https://github.com/NorthlandPositronics/Cogtrix/commit/bd21af79a37622168413466b4be180fd94d8a0d0))
- add default-value assertions to config_safe_int tests (issue [#757](https://github.com/NorthlandPositronics/Cogtrix/issues/757)) ([29cb0a2](https://github.com/NorthlandPositronics/Cogtrix/commit/29cb0a28127207b0072416ce009f040ae6a8549c))
- add explanatory comments for symlink-safe path resolution in \_build_system_prompt (issue [#818](https://github.com/NorthlandPositronics/Cogtrix/issues/818)) ([fa201fe](https://github.com/NorthlandPositronics/Cogtrix/commit/fa201fe704aaded867d69304b968c03be9508f14))
- add provider error recovery with retry/backoff (issue [#657](https://github.com/NorthlandPositronics/Cogtrix/issues/657)) ([2e744cb](https://github.com/NorthlandPositronics/Cogtrix/commit/2e744cb3c26795cc11cf5c27cbb9e4026080cc23))
- add Redis connectivity check to readiness endpoint (issue [#602](https://github.com/NorthlandPositronics/Cogtrix/issues/602)) ([#649](https://github.com/NorthlandPositronics/Cogtrix/issues/649)) ([7dff1c6](https://github.com/NorthlandPositronics/Cogtrix/commit/7dff1c654431458cdc992d694e4b661a31528e7d))
- add tie-breaking to fuzzy tool resolver (issue [#656](https://github.com/NorthlandPositronics/Cogtrix/issues/656)) ([#678](https://github.com/NorthlandPositronics/Cogtrix/issues/678)) ([5c2c082](https://github.com/NorthlandPositronics/Cogtrix/commit/5c2c08261793fa710e1fc1731381d24309dbbd7b))
- address QA review on PR [#701](https://github.com/NorthlandPositronics/Cogtrix/issues/701) — add missing db.commit() in admin routes, fix audit note ([65ae823](https://github.com/NorthlandPositronics/Cogtrix/commit/65ae823bb7f991786a59045237a01e59a82f3b7a))
- address QA review on PR [#701](https://github.com/NorthlandPositronics/Cogtrix/issues/701) (issue [#618](https://github.com/NorthlandPositronics/Cogtrix/issues/618)) ([4b4ab10](https://github.com/NorthlandPositronics/Cogtrix/commit/4b4ab1038ed71052dfce96b604cd7bc08436804b))
- **agent/safety:** add regression test for UserCancelledRun propagation (closes [#1193](https://github.com/NorthlandPositronics/Cogtrix/issues/1193)) ([29bfaba](https://github.com/NorthlandPositronics/Cogtrix/commit/29bfaba19ae5bbc6627464af08abd1d11831ab39))
- **agent/safety:** guard run_confirmation_prompt against None from read_choice ([7ffc92c](https://github.com/NorthlandPositronics/Cogtrix/commit/7ffc92c67216a326c0eef91646d433660a633636))
- **agent/safety:** guard run_confirmation_prompt against None from read_choice ([99a5734](https://github.com/NorthlandPositronics/Cogtrix/commit/99a57348c0de7dfacf2bf659297249a13799bc51)), closes [#1194](https://github.com/NorthlandPositronics/Cogtrix/issues/1194)
- **agent/safety:** guard run_confirmation_prompt against None from read_choice ([#1201](https://github.com/NorthlandPositronics/Cogtrix/issues/1201)) ([2bb8464](https://github.com/NorthlandPositronics/Cogtrix/commit/2bb8464e29b5a4cc411a9abfdf8e1f21941a46c1)), closes [#1194](https://github.com/NorthlandPositronics/Cogtrix/issues/1194)
- **agent/safety:** replace per-thread confirmation lock with shared lock (closes [#1001](https://github.com/NorthlandPositronics/Cogtrix/issues/1001)) ([#1013](https://github.com/NorthlandPositronics/Cogtrix/issues/1013)) ([1f61e40](https://github.com/NorthlandPositronics/Cogtrix/commit/1f61e40105d73ccfb5ac2d50dd0e3c4aa402db91))
- **agent:** deduplicate confirmation block in safety.py ([a8a9bad](https://github.com/NorthlandPositronics/Cogtrix/commit/a8a9badaaa9994166d9ec30634c7cec6ad798608))
- **agent:** guard \_truncate_content against non-positive max_tokens (issue [#801](https://github.com/NorthlandPositronics/Cogtrix/issues/801)) ([319271a](https://github.com/NorthlandPositronics/Cogtrix/commit/319271ad5f9eb4291b61304c1fa50420ce232e37))
- **agent:** guard tool calls against LLM token-limit truncation ([be9c6e1](https://github.com/NorthlandPositronics/Cogtrix/commit/be9c6e12f8c270255ab0088fcd6492ec15059616))
- **agent:** guard tool calls against LLM token-limit truncation ([69254b1](https://github.com/NorthlandPositronics/Cogtrix/commit/69254b11847ac1f7a5abd8b008746c1ef6a9e0af))
- **agent:** re-raise UserCancelledRun in safety.py:403 ([6f2df31](https://github.com/NorthlandPositronics/Cogtrix/commit/6f2df31435612f2441f6aa73d036b7f9dbb439d6))
- **agent:** re-raise UserCancelledRun in safety.py:403 ([d804276](https://github.com/NorthlandPositronics/Cogtrix/commit/d804276c138b0f7b9e1e473bd73dc939dcaf8044))
- **analysis:** track per-search position in \_compute_wshr (closes [#1011](https://github.com/NorthlandPositronics/Cogtrix/issues/1011)) ([#1027](https://github.com/NorthlandPositronics/Cogtrix/issues/1027)) ([99f8c58](https://github.com/NorthlandPositronics/Cogtrix/commit/99f8c585d436311ebba3dcab685a46c811ce58ba))
- **api/auth:** add per-route rate limiting to /register ([#949](https://github.com/NorthlandPositronics/Cogtrix/issues/949)) ([dd56705](https://github.com/NorthlandPositronics/Cogtrix/commit/dd567052347dc13f2bc5de43f77a4a96d5600156))
- **api/auth:** wrap blocking OIDC validate() in asyncio.to_thread (closes [#979](https://github.com/NorthlandPositronics/Cogtrix/issues/979), PR [#1024](https://github.com/NorthlandPositronics/Cogtrix/issues/1024)) ([#1051](https://github.com/NorthlandPositronics/Cogtrix/issues/1051)) ([8a1f1ac](https://github.com/NorthlandPositronics/Cogtrix/commit/8a1f1ac19758f75628194eda8c25dc45bd5d4038))
- **api:** add DB unique constraint on teams(org_id, name) and catch IntegrityError ([#1257](https://github.com/NorthlandPositronics/Cogtrix/issues/1257)) ([64378a0](https://github.com/NorthlandPositronics/Cogtrix/commit/64378a0f8b4f558e6b96506e3d9922136380918a)), closes [#1127](https://github.com/NorthlandPositronics/Cogtrix/issues/1127)
- **api:** add periodic eviction of stale \_hit_counters keys to prevent unbounded memory growth (closes [#958](https://github.com/NorthlandPositronics/Cogtrix/issues/958)) ([#999](https://github.com/NorthlandPositronics/Cogtrix/issues/999)) ([5e0033c](https://github.com/NorthlandPositronics/Cogtrix/commit/5e0033cad369f5b8c8f701bb5b1ba1131495e32b))
- **api:** capture COGTRIX_DB_URL at module import to restore test isolation ([5e3c319](https://github.com/NorthlandPositronics/Cogtrix/commit/5e3c31956edad19c958ee42fad7f645e20451de5))
- **api:** enforce org-scoped access in assign_org_plan (closes [#1123](https://github.com/NorthlandPositronics/Cogtrix/issues/1123)) ([ac7f652](https://github.com/NorthlandPositronics/Cogtrix/commit/ac7f652c9c28c740fe7f0a73b5d7493710e91fc6))
- **api:** enforce org-scoped access in assign_org_plan with assert_same_org (closes [#1123](https://github.com/NorthlandPositronics/Cogtrix/issues/1123)) ([#1150](https://github.com/NorthlandPositronics/Cogtrix/issues/1150)) ([d48648b](https://github.com/NorthlandPositronics/Cogtrix/commit/d48648bf95733a6c280ffec43447d956c78a28b1))
- **api:** enforce plan API call quota on session creation and message sending (closes [#959](https://github.com/NorthlandPositronics/Cogtrix/issues/959)) ([#997](https://github.com/NorthlandPositronics/Cogtrix/issues/997)) ([71b60e4](https://github.com/NorthlandPositronics/Cogtrix/commit/71b60e43074f0e37da1c91075ac466345ea7568b))
- **api:** guard wizard sessions dict with lock and fix TOCTOU in lock init ([15860c3](https://github.com/NorthlandPositronics/Cogtrix/commit/15860c31b60b770b2a58f41aaf9bdff80646a734))
- **api:** guard wizard sessions dict with lock and fix TOCTOU in lock init ([#1209](https://github.com/NorthlandPositronics/Cogtrix/issues/1209)) ([116b5ef](https://github.com/NorthlandPositronics/Cogtrix/commit/116b5ef738064a00ce71ed4f32027dafd6140bfb))
- **api:** honour is_active in login and refresh endpoints ([#122](https://github.com/NorthlandPositronics/Cogtrix/issues/122)) ([f3e3955](https://github.com/NorthlandPositronics/Cogtrix/commit/f3e395563226df4cf354170e459bfd2d5153d46b))
- **api:** honour is_active in login and refresh endpoints ([#122](https://github.com/NorthlandPositronics/Cogtrix/issues/122)) ([d9ca004](https://github.com/NorthlandPositronics/Cogtrix/commit/d9ca004d6a28b098f858e56c2339a091b1afe644))
- **api:** prevent duplicate Stripe customer creation via SELECT FOR UPDATE ([e196aa0](https://github.com/NorthlandPositronics/Cogtrix/commit/e196aa0ea4e114df46b826e8c0458e1937c9030e))
- **api:** remove 5s timeout on done message enqueue to prevent silent drop ([#1252](https://github.com/NorthlandPositronics/Cogtrix/issues/1252)) ([a0b3b6e](https://github.com/NorthlandPositronics/Cogtrix/commit/a0b3b6ecf3403f68fc550774a9d1661c9af04266))
- **api:** remove reentrant lock acquisition in \_wizard_save to prevent deadlock ([54d033b](https://github.com/NorthlandPositronics/Cogtrix/commit/54d033bada91bd52e0f482b2961062629a57f245)), closes [#1197](https://github.com/NorthlandPositronics/Cogtrix/issues/1197)
- **api:** remove unused \_remove_expired_wizard helper ([8852a13](https://github.com/NorthlandPositronics/Cogtrix/commit/8852a132688add0788420bf1f6dc78efaf38aac0))
- **api:** remove WebSocket auth token query parameter (closes [#1128](https://github.com/NorthlandPositronics/Cogtrix/issues/1128)) ([5db74da](https://github.com/NorthlandPositronics/Cogtrix/commit/5db74da014216958a8d175e4391837f4e5ee98be))
- **api:** rename deprecated Starlette HTTP status constants ([d92fcf9](https://github.com/NorthlandPositronics/Cogtrix/commit/d92fcf956da6e690f822145c41932a68fdaba5c0))
- **api:** replace direct approvals mutations with lock-protected methods ([0e9d87b](https://github.com/NorthlandPositronics/Cogtrix/commit/0e9d87bd84ac6a39cdcb930ba2144b288f481291))
- **api:** replace direct approvals mutations with lock-protected methods ([5cc697b](https://github.com/NorthlandPositronics/Cogtrix/commit/5cc697b9a854e87abfa3dff3c6481d77afa4746a)), closes [#1206](https://github.com/NorthlandPositronics/Cogtrix/issues/1206)
- **api:** repo cleanup + langgraph warning suppression + lazy DB engine ([d8beac2](https://github.com/NorthlandPositronics/Cogtrix/commit/d8beac27c73e7a04ea48b5af2625cee2a7907796))
- **api:** reset DB engine cache on lifespan shutdown ([5ce7d1c](https://github.com/NorthlandPositronics/Cogtrix/commit/5ce7d1cdd38052e9ba44cd23e3a1d10fc2eba485))
- **api:** resolve engine module via sys.modules to defeat parent-attr drift ([4620f9e](https://github.com/NorthlandPositronics/Cogtrix/commit/4620f9e393f6fde827bc750d46c80a221567ff2c))
- **api:** rollback in-memory state on remove_mcp_server persist failure ([#1251](https://github.com/NorthlandPositronics/Cogtrix/issues/1251)) ([2f2052e](https://github.com/NorthlandPositronics/Cogtrix/commit/2f2052ec288200948ac079e749ceb9a6914173a2))
- **api:** suppress langgraph compat warning; defer DB engine init ([7456729](https://github.com/NorthlandPositronics/Cogtrix/commit/7456729fe0261e4b4c85143a82c9c7e971b4601f))
- **api:** use configured pool size for db_pool_max in admin status ([#1112](https://github.com/NorthlandPositronics/Cogtrix/issues/1112)) ([#1200](https://github.com/NorthlandPositronics/Cogtrix/issues/1200)) ([78946d1](https://github.com/NorthlandPositronics/Cogtrix/commit/78946d13ccb50bfe2324abcf645f894104427c60))
- **api:** use double-checked locking for \_get_provider_write_lock() ([#1207](https://github.com/NorthlandPositronics/Cogtrix/issues/1207)) ([31303cf](https://github.com/NorthlandPositronics/Cogtrix/commit/31303cf04a71d71ac280b329189a278a64b81d28))
- **api:** use double-checked locking for \_get_provider_write_lock() (closes [#1196](https://github.com/NorthlandPositronics/Cogtrix/issues/1196)) ([3802585](https://github.com/NorthlandPositronics/Cogtrix/commit/3802585a27ba66e50d3938fbb163ab579f99843a))
- **arch:** break bidirectional agent↔orchestration dependency ([#230](https://github.com/NorthlandPositronics/Cogtrix/issues/230)) ([31d6968](https://github.com/NorthlandPositronics/Cogtrix/commit/31d69689c417ac0d839e428e264925e741dba27c))
- **arch:** break bidirectional agent↔orchestration dependency ([#230](https://github.com/NorthlandPositronics/Cogtrix/issues/230)) ([d6a4b5e](https://github.com/NorthlandPositronics/Cogtrix/commit/d6a4b5e858711d84d6b602427ac15b95b90894c2))
- **assistant:** address TOCTOU in DeferralManager save operations (closes [#907](https://github.com/NorthlandPositronics/Cogtrix/issues/907)) ([#1058](https://github.com/NorthlandPositronics/Cogtrix/issues/1058)) ([15c0e92](https://github.com/NorthlandPositronics/Cogtrix/commit/15c0e92c3dbab4594c5048be2eed111cbda1d701))
- **assistant:** clamp \_max_response_length to minimum of 3 (closes [#826](https://github.com/NorthlandPositronics/Cogtrix/issues/826)) ([#880](https://github.com/NorthlandPositronics/Cogtrix/issues/880)) ([c041a6f](https://github.com/NorthlandPositronics/Cogtrix/commit/c041a6f55f78159d65575dbdb61e4be03f5b7d88))
- **assistant:** configure_compression private attr visibility ([#1254](https://github.com/NorthlandPositronics/Cogtrix/issues/1254)) ([7b1051a](https://github.com/NorthlandPositronics/Cogtrix/commit/7b1051a4f9f9c25bf50b85448868ff94f121a706))
- **assistant:** defer \_seeded until channels discovered in cold-start ([5b80af2](https://github.com/NorthlandPositronics/Cogtrix/commit/5b80af286acd3d7be67a9ea358760ecdc9dea7c5)), closes [#1182](https://github.com/NorthlandPositronics/Cogtrix/issues/1182)
- **assistant:** defer \_seeded until channels discovered in cold-start ([#1231](https://github.com/NorthlandPositronics/Cogtrix/issues/1231)) ([51c6055](https://github.com/NorthlandPositronics/Cogtrix/commit/51c60558f57224aa89d4dfdb889e8cfe0cdbd218))
- **assistant:** discard prerecord on all early-return paths in MessageHandler.handle() ([#852](https://github.com/NorthlandPositronics/Cogtrix/issues/852)) ([b0b456b](https://github.com/NorthlandPositronics/Cogtrix/commit/b0b456be29c5c5fdaed3507d4aa95f1c69e4008c)), closes [#850](https://github.com/NorthlandPositronics/Cogtrix/issues/850)
- **assistant:** enforce path containment when data_dir=None ([41caf1d](https://github.com/NorthlandPositronics/Cogtrix/commit/41caf1de618a2c94faa405ed401be704da447b32))
- **assistant:** enforce path containment when data_dir=None ([2b4f07b](https://github.com/NorthlandPositronics/Cogtrix/commit/2b4f07b7ee3053540e5b801ce93ac98a9a245de3)), closes [#1093](https://github.com/NorthlandPositronics/Cogtrix/issues/1093)
- **assistant:** enforce path containment when data_dir=None ([#1237](https://github.com/NorthlandPositronics/Cogtrix/issues/1237)) ([5730cbe](https://github.com/NorthlandPositronics/Cogtrix/commit/5730cbe82b4880ff6027085e9282a805a1075a2c))
- **assistant:** enforce path containment when data_dir=None ([#1256](https://github.com/NorthlandPositronics/Cogtrix/issues/1256)) ([41caf1d](https://github.com/NorthlandPositronics/Cogtrix/commit/41caf1de618a2c94faa405ed401be704da447b32))
- **assistant:** flush FAISS index on shutdown ([#859](https://github.com/NorthlandPositronics/Cogtrix/issues/859)) ([99eef71](https://github.com/NorthlandPositronics/Cogtrix/commit/99eef715c036ca225bbba4f377224241fd0645d5))
- **assistant:** handle list/dict content blocks in datamark_history ([#858](https://github.com/NorthlandPositronics/Cogtrix/issues/858)) ([1a8c768](https://github.com/NorthlandPositronics/Cogtrix/commit/1a8c7686764a076265db9750246ad120152f6f61))
- **assistant:** handle non-array LLM responses in \_extract_facts (closes [#828](https://github.com/NorthlandPositronics/Cogtrix/issues/828)) ([#898](https://github.com/NorthlandPositronics/Cogtrix/issues/898)) ([3c3650d](https://github.com/NorthlandPositronics/Cogtrix/commit/3c3650d6a72f8277f269a150a542b9af9fe02e08))
- **assistant:** harden fact extraction against prompt injection ([#433](https://github.com/NorthlandPositronics/Cogtrix/issues/433)) ([1e47436](https://github.com/NorthlandPositronics/Cogtrix/commit/1e47436c7a67beb687d6c3b97a6db14d3db3522f))
- **assistant:** include exception type in handler error response (closes [#932](https://github.com/NorthlandPositronics/Cogtrix/issues/932)) ([9621f18](https://github.com/NorthlandPositronics/Cogtrix/commit/9621f181a0f3cfade5ba94880398e12a935f541d))
- **assistant:** knowledge.py score_threshold gating ([#388](https://github.com/NorthlandPositronics/Cogtrix/issues/388)) ([851ba26](https://github.com/NorthlandPositronics/Cogtrix/commit/851ba26a8b96dde79a3d2938e424daa4ec270fd0))
- **assistant:** knowledge.py score_threshold gating ([#388](https://github.com/NorthlandPositronics/Cogtrix/issues/388)) ([5876632](https://github.com/NorthlandPositronics/Cogtrix/commit/587663260ee49e3ea1fac6372597f2f916cc8ae5))
- **assistant:** normalize metadata in DeferralManager.\_msg_to_dict (closes [#817](https://github.com/NorthlandPositronics/Cogtrix/issues/817)) ([#857](https://github.com/NorthlandPositronics/Cogtrix/issues/857)) ([cf626c6](https://github.com/NorthlandPositronics/Cogtrix/commit/cf626c6f9882307824fed0e278fc141195e4ab12))
- **assistant:** reset target to pending on launch send failure (closes [#853](https://github.com/NorthlandPositronics/Cogtrix/issues/853)) ([#862](https://github.com/NorthlandPositronics/Cogtrix/issues/862)) ([f8f5051](https://github.com/NorthlandPositronics/Cogtrix/commit/f8f505142831a4e1bd07d89d3d9f9a87f16f95e0))
- **assistant:** return None for empty query in KnowledgeStore.recall (closes [#824](https://github.com/NorthlandPositronics/Cogtrix/issues/824)) ([#879](https://github.com/NorthlandPositronics/Cogtrix/issues/879)) ([96eda96](https://github.com/NorthlandPositronics/Cogtrix/commit/96eda968724d087eb98f5790c64fbdab0677a53c))
- **assistant:** stop fabricated PR references in outbound updates ([#574](https://github.com/NorthlandPositronics/Cogtrix/issues/574)) ([793fa0c](https://github.com/NorthlandPositronics/Cogtrix/commit/793fa0cd9a7305056b03a60c37be565c03073941))
- **assistant:** update last_activity on existing session retrieval ([#909](https://github.com/NorthlandPositronics/Cogtrix/issues/909)) ([#946](https://github.com/NorthlandPositronics/Cogtrix/issues/946)) ([ee87222](https://github.com/NorthlandPositronics/Cogtrix/commit/ee87222b65e716b6e2b44cd4c24c5f4163f65786))
- **assistant:** use **dataclass_fields** in CampaignManager.update() (closes [#854](https://github.com/NorthlandPositronics/Cogtrix/issues/854)) ([#865](https://github.com/NorthlandPositronics/Cogtrix/issues/865)) ([951a6ed](https://github.com/NorthlandPositronics/Cogtrix/commit/951a6ed1602efaa28d6c56f1bb3082edf06588a7))
- **assistant:** warn when \_load_prompt_value reads empty file ([#856](https://github.com/NorthlandPositronics/Cogtrix/issues/856)) ([058b73b](https://github.com/NorthlandPositronics/Cogtrix/commit/058b73b49cd61909c030eaa3ab3edb43ce2e0674))
- **assistant:** wrap subsystem saves in try/except to prevent shutdown hang ([#904](https://github.com/NorthlandPositronics/Cogtrix/issues/904)) ([9ac7591](https://github.com/NorthlandPositronics/Cogtrix/commit/9ac7591f4971a0d4c222418bf18aa94d165b7701))
- **auth:** superadmin role now passes is_admin check (closes [#899](https://github.com/NorthlandPositronics/Cogtrix/issues/899)) ([#939](https://github.com/NorthlandPositronics/Cogtrix/issues/939)) ([4ae7681](https://github.com/NorthlandPositronics/Cogtrix/commit/4ae768150ced34e0871aa6018560603ab7e9f048))
- avoid duplicate extend_run injection ([7e431bc](https://github.com/NorthlandPositronics/Cogtrix/commit/7e431bc816a4803b47988417759d67246ac986c9))
- avoid duplicate extend_run injection ([92180d0](https://github.com/NorthlandPositronics/Cogtrix/commit/92180d0c063d253e93feb01501b553b5f37b43e7))
- block agent turns on WS quota check failure (issue [#666](https://github.com/NorthlandPositronics/Cogtrix/issues/666)) ([#672](https://github.com/NorthlandPositronics/Cogtrix/issues/672)) ([61d6750](https://github.com/NorthlandPositronics/Cogtrix/commit/61d675014b2e4be9734834d69b1d257d19c26cbc))
- block symlink path traversal attacks in file operations (issue [#744](https://github.com/NorthlandPositronics/Cogtrix/issues/744)) ([20fee7b](https://github.com/NorthlandPositronics/Cogtrix/commit/20fee7b19a43688bd50ec14389a2cf1010d28f2a))
- bug queue [#524](https://github.com/NorthlandPositronics/Cogtrix/issues/524) [#525](https://github.com/NorthlandPositronics/Cogtrix/issues/525) [#523](https://github.com/NorthlandPositronics/Cogtrix/issues/523) [#528](https://github.com/NorthlandPositronics/Cogtrix/issues/528) — checkpoint eviction, cron dup, remaps, ISO 8601 ([#542](https://github.com/NorthlandPositronics/Cogtrix/issues/542))
  ([1120dfe](https://github.com/NorthlandPositronics/Cogtrix/commit/1120dfe8db0428d00d58150fe2253847a41f3552))
- **call_model:** make model_max_tokens authoritative; suppress heuristic below cap ([51b16e2](https://github.com/NorthlandPositronics/Cogtrix/commit/51b16e275375b9b2b9fbd95cf6902f07b59dc931))
- catch OSError/PermissionError in memory shutdown + fix compression test assertions (issues [#762](https://github.com/NorthlandPositronics/Cogtrix/issues/762), [#763](https://github.com/NorthlandPositronics/Cogtrix/issues/763)) ([94108e6](https://github.com/NorthlandPositronics/Cogtrix/commit/94108e6d0c6fcdd936d090fc1657a1da0388e6e4))
- **checkpoint:** reject empty/whitespace-only findings ([#533](https://github.com/NorthlandPositronics/Cogtrix/issues/533)) ([0205996](https://github.com/NorthlandPositronics/Cogtrix/commit/020599688b486e58c2b53d0af0520f545d2478b2)), closes [#522](https://github.com/NorthlandPositronics/Cogtrix/issues/522)
- **ci/release:** realign release manifest, fix CODEOWNERS, clean up CI workflows (closes [#1043](https://github.com/NorthlandPositronics/Cogtrix/issues/1043)) ([583450f](https://github.com/NorthlandPositronics/Cogtrix/commit/583450fb13be707a82fbcd7ef42463153b932f89))
- **ci:** fix resource names and API contract in k8s smoke test ([8b4e73c](https://github.com/NorthlandPositronics/Cogtrix/commit/8b4e73c1e540d60b284103450133719c926bdd95))
- **ci:** increase test timeout from 25m to 45m ([#663](https://github.com/NorthlandPositronics/Cogtrix/issues/663)) ([1b61223](https://github.com/NorthlandPositronics/Cogtrix/commit/1b612232d9dffbf6854a9e5eb673c06f6edcd8be))
- **ci:** make Gate 2 cancelled status block merge; revert temp=0.0 pin ([#1277](https://github.com/NorthlandPositronics/Cogtrix/issues/1277)) ([aea1713](https://github.com/NorthlandPositronics/Cogtrix/commit/aea1713701c5708c5c345088ccf7b5d2349aad8d))
- **ci:** reliable test gates, docs-only PR unblocking, SIGTERM handler isolation ([#1247](https://github.com/NorthlandPositronics/Cogtrix/issues/1247)) ([80263c7](https://github.com/NorthlandPositronics/Cogtrix/commit/80263c740858ab4035567a032d028969c8cceeec))
- **ci:** remove Codecov upload ([#988](https://github.com/NorthlandPositronics/Cogtrix/issues/988)) ([be2e35f](https://github.com/NorthlandPositronics/Cogtrix/commit/be2e35f7b49c330cb41ec32f49ba8810cb731b7a))
- **ci:** update docker smoke test to match current API contract ([a58c59f](https://github.com/NorthlandPositronics/Cogtrix/commit/a58c59fb3f8fe2f66c6216ebe6dd09289f28547d))
- **ci:** update docker smoke test to match current API contract ([ac3934b](https://github.com/NorthlandPositronics/Cogtrix/commit/ac3934b292a58386b72ee1db18e2ef9ce45e5cb9))
- **ci:** update k8s deploy smoke test to match current API contract ([289a238](https://github.com/NorthlandPositronics/Cogtrix/commit/289a2383c18dda44537cfff9dc183e479fa34e21))
- **ci:** update k8s deploy smoke test to match current API contract ([98040c2](https://github.com/NorthlandPositronics/Cogtrix/commit/98040c20fa346e42e1f88e853909c5129b5f0701))
- **ci:** upgrade langchain-core from 1.3.2 to 1.3.3 to resolve CVE-2026-44843 ([#1326](https://github.com/NorthlandPositronics/Cogtrix/issues/1326)) ([90c741e](https://github.com/NorthlandPositronics/Cogtrix/commit/90c741ea37601deac58ecefddb68aef388226095))
- **cli:** add null/empty handling for banner config and improve embedding config error message (closes [#803](https://github.com/NorthlandPositronics/Cogtrix/issues/803)) ([#873](https://github.com/NorthlandPositronics/Cogtrix/issues/873)) ([b4394bc](https://github.com/NorthlandPositronics/Cogtrix/commit/b4394bc9253a73641f2c2cc172dfa0aa06d9f2f4))
- **config:** prevent path traversal bypass in resolve_data_path via data/ prefix (fixes [#799](https://github.com/NorthlandPositronics/Cogtrix/issues/799)) ([#867](https://github.com/NorthlandPositronics/Cogtrix/issues/867)) ([0ec162c](https://github.com/NorthlandPositronics/Cogtrix/commit/0ec162cf87d6a2874f15b0772fb7828785a30db7))
- convert inspect.getsource() tests to behavioral tests in test_self_improving_loop_features.py ([#492](https://github.com/NorthlandPositronics/Cogtrix/issues/492)-batch2) ([#558](https://github.com/NorthlandPositronics/Cogtrix/issues/558)) ([5333d4d](https://github.com/NorthlandPositronics/Cogtrix/commit/5333d4d41666a0b82681fac16c7f8e6318807347))
- correct recovery message in handle_fabrication node ([#563](https://github.com/NorthlandPositronics/Cogtrix/issues/563)) ([10eb6f9](https://github.com/NorthlandPositronics/Cogtrix/commit/10eb6f9d0964a4aa5d6010826237d10039b76531))
- correct TOCTOU mitigation — check is_symlink() before resolve() (issue [#808](https://github.com/NorthlandPositronics/Cogtrix/issues/808)) ([78f91c6](https://github.com/NorthlandPositronics/Cogtrix/commit/78f91c694da62d1608fc1908ca083c25017087f6))
- **cron:** scope list/add/remove to calling session — tenant isolation ([f21f3fb](https://github.com/NorthlandPositronics/Cogtrix/commit/f21f3fb75a674be03e5b26d2d450834f64084b27))
- **cron:** scope list/add/remove to calling session — tenant isolation ([#424](https://github.com/NorthlandPositronics/Cogtrix/issues/424)) ([010c93c](https://github.com/NorthlandPositronics/Cogtrix/commit/010c93c8550b16c354cc79c702050e9f934906b1))
- **cron:** wrap llm.invoke in ThreadPoolExecutor with configurable timeout ([#488](https://github.com/NorthlandPositronics/Cogtrix/issues/488)) ([#497](https://github.com/NorthlandPositronics/Cogtrix/issues/497)) ([9a26f9e](https://github.com/NorthlandPositronics/Cogtrix/commit/9a26f9ef9f245a76757c243bca10e19372697108))
- dedupe slack status posts ([03a2f05](https://github.com/NorthlandPositronics/Cogtrix/commit/03a2f055a4ff39464f4dca2f1229160a21db30aa))
- DeepSeek silent exception logging and API key masking (issues [#727](https://github.com/NorthlandPositronics/Cogtrix/issues/727), [#728](https://github.com/NorthlandPositronics/Cogtrix/issues/728)) ([#729](https://github.com/NorthlandPositronics/Cogtrix/issues/729)) ([792c60f](https://github.com/NorthlandPositronics/Cogtrix/commit/792c60ff41b09eacfc6c7c83b6ec920ebfadb08c))
- detect PR references without space (PR[#123](https://github.com/NorthlandPositronics/Cogtrix/issues/123)) in validation regex ([#642](https://github.com/NorthlandPositronics/Cogtrix/issues/642)) ([60a6312](https://github.com/NorthlandPositronics/Cogtrix/commit/60a63125f7aa756bd890790a63350826b757490e))
- detect topic switches and reset session summary ([24c1174](https://github.com/NorthlandPositronics/Cogtrix/commit/24c117406d2ab9b5785ec04a5a0b8103a7328eed))
- detect topic switches and reset session summary ([2e6dec4](https://github.com/NorthlandPositronics/Cogtrix/commit/2e6dec4f9f8a1e52a753d66daa4a541d6087687f))
- **docker:** add rag extra to both Dockerfiles to include faiss-cpu ([4cd201a](https://github.com/NorthlandPositronics/Cogtrix/commit/4cd201a0de06b73e3371b8f9114aac191cab9afa)), closes [#475](https://github.com/NorthlandPositronics/Cogtrix/issues/475)
- **docker:** add rag extra to Docker images to include faiss-cpu ([#475](https://github.com/NorthlandPositronics/Cogtrix/issues/475)) ([f3b719b](https://github.com/NorthlandPositronics/Cogtrix/commit/f3b719b5b4baeb6b12d948e123f027c5511040eb))
- **docker:** cleanup dead code, add bake target, mkdir /data/documents, remove compose healthcheck ([#464](https://github.com/NorthlandPositronics/Cogtrix/issues/464)) ([cd4293f](https://github.com/NorthlandPositronics/Cogtrix/commit/cd4293f719e28708ee9b06511529bdb7f0d5d6aa))
- **docker:** cleanup dead code, add bake target, mkdir /data/documents, remove compose healthcheck ([#464](https://github.com/NorthlandPositronics/Cogtrix/issues/464)) ([6558537](https://github.com/NorthlandPositronics/Cogtrix/commit/65585374bf3f1c521508bbeaf4848236b7caeeb1))
- **docker:** DinD 4 small fixes — uv cache, dirs, docs COPY, XDG path ([d077693](https://github.com/NorthlandPositronics/Cogtrix/commit/d077693d6113fba5b9247718f269664d0db22203))
- **docker:** DinD 4 small fixes — uv cache, dirs, docs COPY, XDG path ([#447](https://github.com/NorthlandPositronics/Cogtrix/issues/447)) ([0a3beff](https://github.com/NorthlandPositronics/Cogtrix/commit/0a3beff1dd5bb62a344aae03cb71708b6cca39f4))
- **docker:** DinD write-path separator + test expectations for /data ([dca9e30](https://github.com/NorthlandPositronics/Cogtrix/commit/dca9e30fe6d74b1f9db5bb94a77ce4e14e9f629c))
- **docker:** DinD write-path separator and test expectations for /data ([#445](https://github.com/NorthlandPositronics/Cogtrix/issues/445), [#446](https://github.com/NorthlandPositronics/Cogtrix/issues/446)) ([c16de7d](https://github.com/NorthlandPositronics/Cogtrix/commit/c16de7d42041c7b4f1fd36fbef04d947c8f3814d))
- **email:** sanitize folder CRLF and validate since date format in search_email (closes [#883](https://github.com/NorthlandPositronics/Cogtrix/issues/883)) ([#1019](https://github.com/NorthlandPositronics/Cogtrix/issues/1019)) ([edeadab](https://github.com/NorthlandPositronics/Cogtrix/commit/edeadabe7bbe97ac486116981b3ec53e32a8c18f))
- enforce agent tool restrictions via tools_include/tools_exclude (issue [#684](https://github.com/NorthlandPositronics/Cogtrix/issues/684)) ([798d55e](https://github.com/NorthlandPositronics/Cogtrix/commit/798d55e0ea4926862ff80017c74560b205c3da2d))
- ensure trimmed context never starts with AIMessage (issue [#742](https://github.com/NorthlandPositronics/Cogtrix/issues/742)) ([d39e28a](https://github.com/NorthlandPositronics/Cogtrix/commit/d39e28ae89113d28b2db6eb9c375155e699b69e0))
- **eval:** disable deepseek-v3 and llama3-70b-cerebras from Gate 2 smoke set ([7400123](https://github.com/NorthlandPositronics/Cogtrix/commit/7400123cb216f40de7d56ba929d5f0abc80628ab))
- **eval:** disable deepseek-v3 and llama3-70b-cerebras from Gate 2 smoke set (issues [#1334](https://github.com/NorthlandPositronics/Cogtrix/issues/1334), [#1332](https://github.com/NorthlandPositronics/Cogtrix/issues/1332)) ([7400123](https://github.com/NorthlandPositronics/Cogtrix/commit/7400123cb216f40de7d56ba929d5f0abc80628ab))
- **eval:** disable deepseek-v3 and llama3-70b-cerebras from Gate 2 smoke set (issues [#1334](https://github.com/NorthlandPositronics/Cogtrix/issues/1334), [#1332](https://github.com/NorthlandPositronics/Cogtrix/issues/1332)) ([c0123b5](https://github.com/NorthlandPositronics/Cogtrix/commit/c0123b5a1dc70bd187927b15538881c746856685))
- **eval:** enrich invoice_approval_workflow tool descriptions for deepseek-v3 ([43603e5](https://github.com/NorthlandPositronics/Cogtrix/commit/43603e534434e46fcfc52b70c25933e556a65f4b))
- **eval:** enrich invoice_approval_workflow tool descriptions for deepseek-v3 ([8fc9173](https://github.com/NorthlandPositronics/Cogtrix/commit/8fc9173629d8cf04535c360a953222819f0811c8)), closes [#1268](https://github.com/NorthlandPositronics/Cogtrix/issues/1268)
- **eval:** enrich invoice_approval_workflow tool descriptions for deepseek-v3 ([#1269](https://github.com/NorthlandPositronics/Cogtrix/issues/1269)) ([43603e5](https://github.com/NorthlandPositronics/Cogtrix/commit/43603e534434e46fcfc52b70c25933e556a65f4b))
- **eval:** enrich procurement supplier_registration tool descriptions for deepseek-v3 ([d9bde11](https://github.com/NorthlandPositronics/Cogtrix/commit/d9bde112e6bc41d3fa9a80bdfa97e86eec14ff4f))
- **eval:** enrich procurement supplier_registration tool descriptions for deepseek-v3 ([eed8833](https://github.com/NorthlandPositronics/Cogtrix/commit/eed88331cd9bc4bca7f2a557252190acfd3bfae5))
- **eval:** enrich procurement supplier_registration tool descriptions for deepseek-v3 ([#1275](https://github.com/NorthlandPositronics/Cogtrix/issues/1275)) ([d9bde11](https://github.com/NorthlandPositronics/Cogtrix/commit/d9bde112e6bc41d3fa9a80bdfa97e86eec14ff4f))
- **eval:** raise recovery_synthesis budget so claude-sonnet stays under ceiling ([2cd8941](https://github.com/NorthlandPositronics/Cogtrix/commit/2cd89415c89da06dbe31715b7ccebf7fe3722e25))
- expand \_check_exfiltration to handle tuples, sets, dicts and other iterables (issue [#819](https://github.com/NorthlandPositronics/Cogtrix/issues/819)) ([c74a18b](https://github.com/NorthlandPositronics/Cogtrix/commit/c74a18bb10e0edf90ea7975fc2ee9a2d29145d2b))
- extend \_ABS_PATH_RE to match single-segment absolute paths (issue [#737](https://github.com/NorthlandPositronics/Cogtrix/issues/737)) ([#750](https://github.com/NorthlandPositronics/Cogtrix/issues/750)) ([0e2c96c](https://github.com/NorthlandPositronics/Cogtrix/commit/0e2c96cb2abfb2031fade1164e68b6e181052142))
- extend stuck detection to track consecutive tool errors with patterns (ISSUE [#581](https://github.com/NorthlandPositronics/Cogtrix/issues/581)) ([#623](https://github.com/NorthlandPositronics/Cogtrix/issues/623)) ([9ecd072](https://github.com/NorthlandPositronics/Cogtrix/commit/9ecd0721455b1ccdb23c39a726607ef620dd7b46))
- gate merge retries on live CI status ([de89b90](https://github.com/NorthlandPositronics/Cogtrix/commit/de89b9045e49026ebecfde2582ddf59651daf490))
- **gate2:** require structural completion AND judge approval (closes [#1268](https://github.com/NorthlandPositronics/Cogtrix/issues/1268)) ([df8c9d4](https://github.com/NorthlandPositronics/Cogtrix/commit/df8c9d41e24cff27af6692e2a2688d404b736d7f))
- **gate2:** require structural completion AND judge approval (closes [#1268](https://github.com/NorthlandPositronics/Cogtrix/issues/1268)) ([4d0eea4](https://github.com/NorthlandPositronics/Cogtrix/commit/4d0eea4e4e0bb2bb9d7d88ffff6b5dcc4fc9c6a4))
- **gate2:** require structural completion AND judge approval (closes [#1268](https://github.com/NorthlandPositronics/Cogtrix/issues/1268)) ([#1276](https://github.com/NorthlandPositronics/Cogtrix/issues/1276)) ([df8c9d4](https://github.com/NorthlandPositronics/Cogtrix/commit/df8c9d41e24cff27af6692e2a2688d404b736d7f))
- **graph:** detect and break temporal polling loops ([02f8d13](https://github.com/NorthlandPositronics/Cogtrix/commit/02f8d13d7e327f13e33af06ff1ac5f4b85e0bc1b))
- **graph:** detect and break temporal polling loops ([#473](https://github.com/NorthlandPositronics/Cogtrix/issues/473)) ([85fe408](https://github.com/NorthlandPositronics/Cogtrix/commit/85fe408c190cd3f7ac13d21abfc4f54dc803e5b1))
- **graph:** suppress action-intent nudge on access-denied tool failures ([bec9f66](https://github.com/NorthlandPositronics/Cogtrix/commit/bec9f66de4bae2aac65a1c99efc77c8129834640))
- **graph:** suppress action-intent nudge on access-denied tool failures ([#410](https://github.com/NorthlandPositronics/Cogtrix/issues/410)) ([f866620](https://github.com/NorthlandPositronics/Cogtrix/commit/f8666206e6101c3547be7ba548103b771165e483))
- **graph:** topic-switch detects imperatives + quality gate allows empty JSON ([f0aa231](https://github.com/NorthlandPositronics/Cogtrix/commit/f0aa231dea0141ddc2835ad0c3bf278afdd93c4f))
- **graph:** topic-switch detects imperatives + quality gate allows empty JSON ([#417](https://github.com/NorthlandPositronics/Cogtrix/issues/417)) ([9eeeeb0](https://github.com/NorthlandPositronics/Cogtrix/commit/9eeeeb0b2af9206f38472e9291b4192866fe4c2f))
- guard against None values in delegate_task and delegate_parallel (issue [#810](https://github.com/NorthlandPositronics/Cogtrix/issues/810)) ([4a18a57](https://github.com/NorthlandPositronics/Cogtrix/commit/4a18a578e850031399493956b7e87b8939913d8b))
- handle dict-type content blocks in \_content_len and \_msg_tokens (issue [#790](https://github.com/NorthlandPositronics/Cogtrix/issues/790)) ([9573cd1](https://github.com/NorthlandPositronics/Cogtrix/commit/9573cd189875074df92c6572ae74c8bf07750d0f))
- improve telemetry coverage and secret scrubbing (issue [#759](https://github.com/NorthlandPositronics/Cogtrix/issues/759)) ([3d76feb](https://github.com/NorthlandPositronics/Cogtrix/commit/3d76feba4df472f817a9374a1ccc3c66ac8fcb4c))
- **infra:** upgrade mako 1.3.11 -&gt; 1.3.12 and python-multipart 0.0.26 -&gt; 0.0.27 to resolve CI dependency security CVEs ([c3b8620](https://github.com/NorthlandPositronics/Cogtrix/commit/c3b8620d787d735722d1eebac5628d1db97d1471)), closes [#1111](https://github.com/NorthlandPositronics/Cogtrix/issues/1111)
- **infra:** upgrade mako 1.3.11-&gt;1.3.12, python-multipart 0.0.26-&gt;0.0.27 for CI security ([e803db9](https://github.com/NorthlandPositronics/Cogtrix/commit/e803db9099709e6ed765605d6d06667a2322d2dd))
- **infra:** upgrade mako 1.3.11-&gt;1.3.12, python-multipart 0.0.26-&gt;0.0.27 for CI security ([#1114](https://github.com/NorthlandPositronics/Cogtrix/issues/1114)) ([e803db9](https://github.com/NorthlandPositronics/Cogtrix/commit/e803db9099709e6ed765605d6d06667a2322d2dd))
- **knowledge:** harden fact extraction prompts ([e69e75c](https://github.com/NorthlandPositronics/Cogtrix/commit/e69e75c795df5b559dce13dbf0d0d0f2f98cf3e4))
- **ldap:** validate search_filter and group_filter — prevent overly broad directory queries ([#429](https://github.com/NorthlandPositronics/Cogtrix/issues/429)) ([1e151d9](https://github.com/NorthlandPositronics/Cogtrix/commit/1e151d968be83ff5926abf6c6c26a9afee3a60de))
- **ldap:** validate search_filter and group_filter — prevent overly broad directory queries ([#429](https://github.com/NorthlandPositronics/Cogtrix/issues/429)) ([7497025](https://github.com/NorthlandPositronics/Cogtrix/commit/7497025c11e9680caa560d3add5f48ae868767b7))
- log malformed DeepSeek URLs and add context to URL validation (issue [#800](https://github.com/NorthlandPositronics/Cogtrix/issues/800)) ([40c56ff](https://github.com/NorthlandPositronics/Cogtrix/commit/40c56ffde551c175e1b2aa42dc3ed92359bf30d2))
- log resolve_llm_config failures instead of silently swallowing them (issue [#831](https://github.com/NorthlandPositronics/Cogtrix/issues/831)) ([f9ea920](https://github.com/NorthlandPositronics/Cogtrix/commit/f9ea920029c91bf227b78936d499a9da027e98de))
- log rollback failures instead of silently swallowing them (issue [#830](https://github.com/NorthlandPositronics/Cogtrix/issues/830)) ([1bb6349](https://github.com/NorthlandPositronics/Cogtrix/commit/1bb6349914893c2b9a194ddf5cbb69d665909228))
- **logging:** target \_KEY_NAME_RE to key-value patterns only ([#994](https://github.com/NorthlandPositronics/Cogtrix/issues/994)) ([af2a0b5](https://github.com/NorthlandPositronics/Cogtrix/commit/af2a0b53b489150abb596a415eae0f8b90b1d1ec))
- make \_confirmation_lock per-thread and reduce lock scope (issue [#736](https://github.com/NorthlandPositronics/Cogtrix/issues/736)) ([#738](https://github.com/NorthlandPositronics/Cogtrix/issues/738)) ([5a93c83](https://github.com/NorthlandPositronics/Cogtrix/commit/5a93c838f70fe1d258d04ddec833e9803c619bf1))
- MCP reconnect timeout to prevent hangs ([#576](https://github.com/NorthlandPositronics/Cogtrix/issues/576)) ([#637](https://github.com/NorthlandPositronics/Cogtrix/issues/637)) ([466c96e](https://github.com/NorthlandPositronics/Cogtrix/commit/466c96e405c754cdd84cba7f635b56c02b215424))
- **mcp:** add allow_insecure to KNOWN_MCP_FIELDS so yaml config reaches MCPServerConfig ([#395](https://github.com/NorthlandPositronics/Cogtrix/issues/395)) ([6492055](https://github.com/NorthlandPositronics/Cogtrix/commit/6492055f88764a6f1c2bcd302b5632202a4dc8da))
- **mcp:** add allow_insecure to KNOWN_MCP_FIELDS so yaml config reaches MCPServerConfig ([#395](https://github.com/NorthlandPositronics/Cogtrix/issues/395)) ([b501e59](https://github.com/NorthlandPositronics/Cogtrix/commit/b501e59faf3b0b79b7fcc923187f9d6f6474b8b7))
- **mcp:** allow_insecure also bypasses RFC1918/private-IP block ([#395](https://github.com/NorthlandPositronics/Cogtrix/issues/395)) ([2703387](https://github.com/NorthlandPositronics/Cogtrix/commit/2703387d8823a7812b29a43d3cd746f534d3ea62))
- **mcp:** annotate renamed tool descriptions ([#526](https://github.com/NorthlandPositronics/Cogtrix/issues/526)) ([#539](https://github.com/NorthlandPositronics/Cogtrix/issues/539)) ([d2ac59b](https://github.com/NorthlandPositronics/Cogtrix/commit/d2ac59b0c18fb1eecaa35274f092bb77ef6c9919))
- **mcp:** auto-prepend /workspace/ to relative paths in filesystem tools ([#527](https://github.com/NorthlandPositronics/Cogtrix/issues/527)) ([#541](https://github.com/NorthlandPositronics/Cogtrix/issues/541)) ([bf7dc27](https://github.com/NorthlandPositronics/Cogtrix/commit/bf7dc27dc8868497090840e6a692126c75fcd59c))
- **mcp:** don't restore stderr in close_all() — silences post_writer traceback ([#500](https://github.com/NorthlandPositronics/Cogtrix/issues/500)) ([#502](https://github.com/NorthlandPositronics/Cogtrix/issues/502)) ([88d4bdc](https://github.com/NorthlandPositronics/Cogtrix/commit/88d4bdc2bb9ea5ea797ffc154b86f3c507dbbdd7))
- **mcp:** guard \_ensure_loop() against post-close_all() zombie loop ([2d547e5](https://github.com/NorthlandPositronics/Cogtrix/commit/2d547e5da09993041b081044a94c41c6fa8c6f5c))
- **mcp:** guard \_ensure_loop() against post-close_all() zombie loop ([#425](https://github.com/NorthlandPositronics/Cogtrix/issues/425)) ([d3f2251](https://github.com/NorthlandPositronics/Cogtrix/commit/d3f2251be5996c072f036515ce4a173b1685f354))
- **mcp:** make filesystem path prefix configurable ([#543](https://github.com/NorthlandPositronics/Cogtrix/issues/543)) ([#557](https://github.com/NorthlandPositronics/Cogtrix/issues/557)) ([8eb611f](https://github.com/NorthlandPositronics/Cogtrix/commit/8eb611f5d61ff281879082e651ccbbca26ee3376))
- **mcp:** pre-close SSE transports before loop stop ([#504](https://github.com/NorthlandPositronics/Cogtrix/issues/504)) ([#534](https://github.com/NorthlandPositronics/Cogtrix/issues/534)) ([6c61839](https://github.com/NorthlandPositronics/Cogtrix/commit/6c61839a1a5120a489653734c60c133249a7bc98))
- **mcp:** remove dead \_old_stderr assignment in close_all() ([#513](https://github.com/NorthlandPositronics/Cogtrix/issues/513)) ([#536](https://github.com/NorthlandPositronics/Cogtrix/issues/536)) ([7443ce1](https://github.com/NorthlandPositronics/Cogtrix/commit/7443ce1b8b8629caa44585233ed1039cd32db90e))
- **mcp:** serialize reconnect via per-server asyncio.Lock — closes race [#427](https://github.com/NorthlandPositronics/Cogtrix/issues/427) ([8eb86f1](https://github.com/NorthlandPositronics/Cogtrix/commit/8eb86f1cca2eeb71d43ef296caf96d4b2c4a3c5f))
- **mcp:** serialize reconnect via per-server asyncio.Lock — closes race [#427](https://github.com/NorthlandPositronics/Cogtrix/issues/427) ([df972ed](https://github.com/NorthlandPositronics/Cogtrix/commit/df972edf9748326404deb555528b331434089a76))
- **mcp:** startup retry + explicit anyio cleanup on connect failure ([420b767](https://github.com/NorthlandPositronics/Cogtrix/commit/420b767f60a2047782b598f7cf39bce7e0ab9c99))
- **mcp:** startup retry + explicit cleanup on connect failure ([#393](https://github.com/NorthlandPositronics/Cogtrix/issues/393), [#403](https://github.com/NorthlandPositronics/Cogtrix/issues/403)) ([9661df4](https://github.com/NorthlandPositronics/Cogtrix/commit/9661df40ac3c0be123a75e4c8e4ad21321f2d02a))
- **mcp:** unify shutdown guard state across close and reconnect ([#546](https://github.com/NorthlandPositronics/Cogtrix/issues/546)) ([#551](https://github.com/NorthlandPositronics/Cogtrix/issues/551)) ([5087765](https://github.com/NorthlandPositronics/Cogtrix/commit/508776572ac3951a370862efd016bba4604967fb))
- **memory:** acquire \_lock in unregister() to prevent race with register/create ([06caf41](https://github.com/NorthlandPositronics/Cogtrix/commit/06caf411add21d369fd8b9e815afe444a75e8355))
- **memory:** acquire \_lock in unregister() to prevent race with register/create ([9bdad7f](https://github.com/NorthlandPositronics/Cogtrix/commit/9bdad7f1d0eb0dbc477a59fa116e248a23348080))
- **memory:** acquire \_lock in unregister() to prevent race with register/create ([#1081](https://github.com/NorthlandPositronics/Cogtrix/issues/1081)) ([06caf41](https://github.com/NorthlandPositronics/Cogtrix/commit/06caf411add21d369fd8b9e815afe444a75e8355))
- **memory:** add \_bg_in_flight guard to \_check_summary_token_ttl to prevent summary race (closes [#1078](https://github.com/NorthlandPositronics/Cogtrix/issues/1078)) ([678cbdd](https://github.com/NorthlandPositronics/Cogtrix/commit/678cbdda00858c518027a8224ab9a0d5d43d7b8f))
- **memory:** add double-checked lock to \_get_facts_store() lazy init (closes [#1187](https://github.com/NorthlandPositronics/Cogtrix/issues/1187)) ([#1195](https://github.com/NorthlandPositronics/Cogtrix/issues/1195)) ([b5cceed](https://github.com/NorthlandPositronics/Cogtrix/commit/b5cceedd913ba470d58219a0fd7297431c6d36ab))
- **memory:** add ThreadPoolExecutor timeout to distill_summary() ([f83b29c](https://github.com/NorthlandPositronics/Cogtrix/commit/f83b29c0784ba08c28d54c4b07235c7838a3ed39))
- **memory:** add ThreadPoolExecutor timeout to distill_summary() ([75f6e7b](https://github.com/NorthlandPositronics/Cogtrix/commit/75f6e7b4b06977b00295aabd7660cc0fff48a4a4)), closes [#1143](https://github.com/NorthlandPositronics/Cogtrix/issues/1143)
- **memory:** add ThreadPoolExecutor timeout to distill_summary() ([#1149](https://github.com/NorthlandPositronics/Cogtrix/issues/1149)) ([f83b29c](https://github.com/NorthlandPositronics/Cogtrix/commit/f83b29c0784ba08c28d54c4b07235c7838a3ed39))
- **memory:** add timeout to compress_to_tier LLM invoke to prevent background thread exhaustion ([b32620c](https://github.com/NorthlandPositronics/Cogtrix/commit/b32620cf76e1954328a30327adc31560b505ceb0))
- **memory:** avoid stale summary ttl reset ([c48459f](https://github.com/NorthlandPositronics/Cogtrix/commit/c48459fe358afa257eeadd0b3c49967bafffe1f4))
- **memory:** avoid stale summary TTL reset ([#434](https://github.com/NorthlandPositronics/Cogtrix/issues/434)) ([d670560](https://github.com/NorthlandPositronics/Cogtrix/commit/d670560f0eb1f44bb07fb13ba14bbabcd6bc6d68))
- **memory:** log embedding init failures at WARNING with one-shot flag ([#1190](https://github.com/NorthlandPositronics/Cogtrix/issues/1190)) ([cbd37a8](https://github.com/NorthlandPositronics/Cogtrix/commit/cbd37a801b7ad80ffbbeaaacb7eaea07314151b0))
- **memory:** make vector_recall_k configurable via env var and config file ([#619](https://github.com/NorthlandPositronics/Cogtrix/issues/619)) ([e88a1c9](https://github.com/NorthlandPositronics/Cogtrix/commit/e88a1c93a887b35acc6ae689484740cf90ab371c))
- **memory:** protect mutable state in modes with \_mode_lock (closes [#902](https://github.com/NorthlandPositronics/Cogtrix/issues/902)) ([#948](https://github.com/NorthlandPositronics/Cogtrix/issues/948)) ([8610aca](https://github.com/NorthlandPositronics/Cogtrix/commit/8610aca5802fcbbc0a4c7be23b964a89a4729dd8))
- **memory:** re-check \_slow_path_failures under lock + persist dirty summary (closes [#963](https://github.com/NorthlandPositronics/Cogtrix/issues/963), [#964](https://github.com/NorthlandPositronics/Cogtrix/issues/964)) ([#1050](https://github.com/NorthlandPositronics/Cogtrix/issues/1050))
  ([732d7be](https://github.com/NorthlandPositronics/Cogtrix/commit/732d7bed5d267b7b6a944c9ff2171a49d18c9a27))
- **memory:** reasoning mode no longer asks for clarification on search/lookup requests ([#892](https://github.com/NorthlandPositronics/Cogtrix/issues/892)) ([b423056](https://github.com/NorthlandPositronics/Cogtrix/commit/b4230562fe15e7b0ec70057a77622f72185e30dc)), closes [#890](https://github.com/NorthlandPositronics/Cogtrix/issues/890)
- **memory:** replace with-ThreadPoolExecutor context manager with manual shutdown(wait=False) in distill_summary ([d853a96](https://github.com/NorthlandPositronics/Cogtrix/commit/d853a96951ed8ec7dfd58fe462a9def0536b1c2f))
- **memory:** reset \_tokens_since_summary after background summarization ([4f68715](https://github.com/NorthlandPositronics/Cogtrix/commit/4f68715361003badb8d4fcc10392f8876fb03406))
- **memory:** reset \_tokens_since_summary after background summarization ([#486](https://github.com/NorthlandPositronics/Cogtrix/issues/486)) ([963833e](https://github.com/NorthlandPositronics/Cogtrix/commit/963833eae4dab7ec87eab22d8a289544b64a28dd))
- **memory:** return None from \_get_hybrid_snapshot when non-blocking lock fails (closes [#1033](https://github.com/NorthlandPositronics/Cogtrix/issues/1033)) ([#1039](https://github.com/NorthlandPositronics/Cogtrix/issues/1039)) ([dfa6edd](https://github.com/NorthlandPositronics/Cogtrix/commit/dfa6eddd11686a6d1302dc5d2301746bad920bc1))
- **memory:** token-budget working memory eviction ([5727ceb](https://github.com/NorthlandPositronics/Cogtrix/commit/5727ceb9cb0c87e82b1365584312cc3f5b5acd44))
- **memory:** wrap generate_summary llm.invoke in ThreadPoolExecutor with 60s timeout (closes [#1129](https://github.com/NorthlandPositronics/Cogtrix/issues/1129)) ([#1142](https://github.com/NorthlandPositronics/Cogtrix/issues/1142)) ([48bd0e4](https://github.com/NorthlandPositronics/Cogtrix/commit/48bd0e42aa24796b22c13fe7c016e112de605611))
- **metrics:** add auth + path normalization to metrics endpoint ([#487](https://github.com/NorthlandPositronics/Cogtrix/issues/487)) ([#501](https://github.com/NorthlandPositronics/Cogtrix/issues/501)) ([7a3727d](https://github.com/NorthlandPositronics/Cogtrix/commit/7a3727d0f31aa14338dbf2fef12567f101ffa9d6))
- narrow exception handling in quiet window functions (issue [#815](https://github.com/NorthlandPositronics/Cogtrix/issues/815)) ([a76d7ca](https://github.com/NorthlandPositronics/Cogtrix/commit/a76d7ca976aa9e344337e2bceb28789fda84501e))
- **orchestration:** acquire \_tool_budget_lock when clearing \_tool_call_counts in \_reset_for_new_run (closes [#864](https://github.com/NorthlandPositronics/Cogtrix/issues/864)) ([#875](https://github.com/NorthlandPositronics/Cogtrix/issues/875)) ([d1a25d5](https://github.com/NorthlandPositronics/Cogtrix/commit/d1a25d59b29b05df7dc39094bd9225af1f7c043b))
- **orchestration:** add 60s timeout to \_classify_think_task llm.invoke (closes [#1132](https://github.com/NorthlandPositronics/Cogtrix/issues/1132)) ([#1139](https://github.com/NorthlandPositronics/Cogtrix/issues/1139)) ([745d27d](https://github.com/NorthlandPositronics/Cogtrix/commit/745d27d55f631998f0b2cfcb91e994b62413d6ad))
- **orchestration:** add lock to SessionOrchestrator snapshot/rollback ([bb1ce85](https://github.com/NorthlandPositronics/Cogtrix/commit/bb1ce852e734082e5015ebbee40321d6ed45e69b))
- **orchestration:** add lock to SessionOrchestrator snapshot/rollback ([855d115](https://github.com/NorthlandPositronics/Cogtrix/commit/855d11518d7bbf206b9d235e361a72b98173dd20))
- **orchestration:** add ThreadPoolExecutor timeouts to prevent **exit** blocking on hung threads ([d119cd4](https://github.com/NorthlandPositronics/Cogtrix/commit/d119cd40427962e9b1767bfff1897405ad3d2a13))
- **orchestration:** add ThreadPoolExecutor timeouts to prevent **exit** blocking on hung threads ([f5e2886](https://github.com/NorthlandPositronics/Cogtrix/commit/f5e2886e908403c54cb07d1e6f873215386c4e3e)), closes [#1158](https://github.com/NorthlandPositronics/Cogtrix/issues/1158)
- **orchestration:** avoid ThreadPoolExecutor **exit** hang in compression timeout paths ([c918b0f](https://github.com/NorthlandPositronics/Cogtrix/commit/c918b0f4bd7ce9bc616bd14ddae42c28f9bbdab4))
- **orchestration:** complete process_tools extraction, bring parity with inline version ([#1191](https://github.com/NorthlandPositronics/Cogtrix/issues/1191)) ([ec6ce27](https://github.com/NorthlandPositronics/Cogtrix/commit/ec6ce275eb946ea02669cb111a5c4c51704c1d4f)), closes [#1159](https://github.com/NorthlandPositronics/Cogtrix/issues/1159)
- **orchestration:** detect fabricated success after tool failures ([#538](https://github.com/NorthlandPositronics/Cogtrix/issues/538)) ([#540](https://github.com/NorthlandPositronics/Cogtrix/issues/540)) ([eae4778](https://github.com/NorthlandPositronics/Cogtrix/commit/eae4778940d8f01df63961c8bdb5fb8f9c2e1808))
- **orchestration:** export TOOL_CALLS_TOTAL from metrics and guard None usage (closes [#1060](https://github.com/NorthlandPositronics/Cogtrix/issues/1060)) ([#1061](https://github.com/NorthlandPositronics/Cogtrix/issues/1061)) ([cee5574](https://github.com/NorthlandPositronics/Cogtrix/commit/cee557410e621d8dc8ad54efab4e5f3327aba678))
- **orchestration:** lock \_same_file_writes and \_checkpoint_store clears in \_reset_for_new_run (closes [#884](https://github.com/NorthlandPositronics/Cogtrix/issues/884)) ([#887](https://github.com/NorthlandPositronics/Cogtrix/issues/887)) ([9525813](https://github.com/NorthlandPositronics/Cogtrix/commit/95258132d9f84c5fc8116d32e207b8d094e4fa81))
- **orchestration:** log warning when \_correct_tool_args schema introspection fails (closes [#863](https://github.com/NorthlandPositronics/Cogtrix/issues/863)) ([#872](https://github.com/NorthlandPositronics/Cogtrix/issues/872)) ([ce412cb](https://github.com/NorthlandPositronics/Cogtrix/commit/ce412cbb047f71a66c5e67c25447573b7066b34a))
- **orchestration:** log warnings when tool \_resolve fails in runner.py (fixes [#861](https://github.com/NorthlandPositronics/Cogtrix/issues/861)) ([#869](https://github.com/NorthlandPositronics/Cogtrix/issues/869)) ([9c38a80](https://github.com/NorthlandPositronics/Cogtrix/commit/9c38a80989fd962f86686e51d888e70259ae2442))
- **orchestration:** migrate run_agent() to AgentRunConfig-only ([#228](https://github.com/NorthlandPositronics/Cogtrix/issues/228)) ([#499](https://github.com/NorthlandPositronics/Cogtrix/issues/499)) ([086a5df](https://github.com/NorthlandPositronics/Cogtrix/commit/086a5df342a175261590dbdd704b9bf831e01e4e))
- **orchestration:** propagate UserCancelledRun through force_deep_think ([8f843ef](https://github.com/NorthlandPositronics/Cogtrix/commit/8f843efc0d13a22c00bef5b00153e4eb322dd81d))
- **orchestration:** propagate UserCancelledRun through force_deep_think ([#1185](https://github.com/NorthlandPositronics/Cogtrix/issues/1185)) ([f42a5f3](https://github.com/NorthlandPositronics/Cogtrix/commit/f42a5f377e0c49cf6633fa1ec19c34baf2ab26ca))
- **orchestration:** protect \_tool_call_counts read in soft budget nudge with \_tool_budget_lock (closes [#951](https://github.com/NorthlandPositronics/Cogtrix/issues/951)) ([#954](https://github.com/NorthlandPositronics/Cogtrix/issues/954)) ([1f7bb75](https://github.com/NorthlandPositronics/Cogtrix/commit/1f7bb75c5c50bed2436c5ae03d532038ccb11b62))
- **orchestration:** protect \_tool_lookup.get() read with \_tool_budget_lock to close TOCTOU race (closes [#961](https://github.com/NorthlandPositronics/Cogtrix/issues/961)) ([#990](https://github.com/NorthlandPositronics/Cogtrix/issues/990)) ([112ae73](https://github.com/NorthlandPositronics/Cogtrix/commit/112ae73912b2a8059523f92029d4a34edd44d237))
- **orchestration:** protect per-session cache merge with config.cache_lock to prevent data race (closes [#956](https://github.com/NorthlandPositronics/Cogtrix/issues/956)) ([#995](https://github.com/NorthlandPositronics/Cogtrix/issues/995)) ([7fbe1c5](https://github.com/NorthlandPositronics/Cogtrix/commit/7fbe1c58b47de1f8bc4472fec5759b1649217814))
- **orchestration:** replace per-call ThreadPoolExecutor with shared bounded pool to prevent thread leak (closes [#957](https://github.com/NorthlandPositronics/Cogtrix/issues/957)) ([#992](https://github.com/NorthlandPositronics/Cogtrix/issues/992)) ([ac46677](https://github.com/NorthlandPositronics/Cogtrix/commit/ac466779e3f22f92ae1427d5ad3d233b23195954))
- **orchestration:** resolve \_classify_query_complexity circular dep (closes [#802](https://github.com/NorthlandPositronics/Cogtrix/issues/802)) ([#878](https://github.com/NorthlandPositronics/Cogtrix/issues/878)) ([432e1ed](https://github.com/NorthlandPositronics/Cogtrix/commit/432e1ed1a27482e2ced6d16fded4922c1ff800c4))
- **orchestration:** scope \_drain_background_compression_jobs to target cache (closes [#901](https://github.com/NorthlandPositronics/Cogtrix/issues/901)) ([#1005](https://github.com/NorthlandPositronics/Cogtrix/issues/1005)) ([f98d5d1](https://github.com/NorthlandPositronics/Cogtrix/commit/f98d5d18cd53588244807a4a85daef75e8e68d9f))
- **orchestration:** scope \_drain_background_compression_jobs to target cache (closes [#901](https://github.com/NorthlandPositronics/Cogtrix/issues/901)) ([#1005](https://github.com/NorthlandPositronics/Cogtrix/issues/1005)) ([#1056](https://github.com/NorthlandPositronics/Cogtrix/issues/1056))
  ([6c1d275](https://github.com/NorthlandPositronics/Cogtrix/commit/6c1d275fbb1804d1b3fc38ceb2fe0cbc4c5dba04))
- **orchestration:** stabilize tool arg schema cache key ([#521](https://github.com/NorthlandPositronics/Cogtrix/issues/521)) ([#532](https://github.com/NorthlandPositronics/Cogtrix/issues/532)) ([342cf1a](https://github.com/NorthlandPositronics/Cogtrix/commit/342cf1ae2757ffb006b00b7ebe56e0fc0c546188))
- **orchestration:** synthesize answer when recovery gives up; add Gate 2 regression test ([#1255](https://github.com/NorthlandPositronics/Cogtrix/issues/1255)) ([89dfb8b](https://github.com/NorthlandPositronics/Cogtrix/commit/89dfb8b2ab99c698a4ec8e3412d1e8355ff79aeb))
- **orchestration:** tool output quality gate — Layer 3 ([#382](https://github.com/NorthlandPositronics/Cogtrix/issues/382)) ([a2b0655](https://github.com/NorthlandPositronics/Cogtrix/commit/a2b06550e35753198c3bc792c821678f6d860ac3))
- **orchestration:** tool output quality gate — Layer 3 ([#382](https://github.com/NorthlandPositronics/Cogtrix/issues/382)) ([e5294b6](https://github.com/NorthlandPositronics/Cogtrix/commit/e5294b69027e233bb2a8ff3204b40a2402bb4689))
- **orchestration:** wrap compression LLM.invoke() with ThreadPoolExecutor timeout to prevent pool exhaustion ([8f242dc](https://github.com/NorthlandPositronics/Cogtrix/commit/8f242dc67663691d870172ebd7acd8ae797938a0))
- **orchestration:** wrap compression LLM.invoke() with ThreadPoolExecutor timeout to prevent pool exhaustion (closes [#1151](https://github.com/NorthlandPositronics/Cogtrix/issues/1151)) ([3ccf149](https://github.com/NorthlandPositronics/Cogtrix/commit/3ccf1492584eaf616dd7e7f566702f6a94ba0183))
- pass score_threshold to FAISS knowledge recall (issue [#655](https://github.com/NorthlandPositronics/Cogtrix/issues/655)) ([200406e](https://github.com/NorthlandPositronics/Cogtrix/commit/200406ee0c649e904fce97b3434b10f8229b8052))
- pin cron tools in reasoning mode ([#564](https://github.com/NorthlandPositronics/Cogtrix/issues/564)) ([8693e48](https://github.com/NorthlandPositronics/Cogtrix/commit/8693e480035acd733012cbee58cd93f176d852bc))
- **pm:** gate merge retries on live CI status ([#160](https://github.com/NorthlandPositronics/Cogtrix/issues/160)) ([34e7717](https://github.com/NorthlandPositronics/Cogtrix/commit/34e77175ad4dc3478b09f63cc3ce6a84321bde1d))
- populate source_session in knowledge store for audit trail (issue [#679](https://github.com/NorthlandPositronics/Cogtrix/issues/679)) ([#686](https://github.com/NorthlandPositronics/Cogtrix/issues/686)) ([8caf7b1](https://github.com/NorthlandPositronics/Cogtrix/commit/8caf7b13e452accf9c7962432ba83feca8fda727))
- preserve duplicate detection for non-serializable args ([2d998dd](https://github.com/NorthlandPositronics/Cogtrix/commit/2d998ddb87b5f25f8e1d83dc16c39da9afa1a56d))
- preserve LangChain list content on memory serialization (issue [#714](https://github.com/NorthlandPositronics/Cogtrix/issues/714)) ([cbfeb91](https://github.com/NorthlandPositronics/Cogtrix/commit/cbfeb9107206ad54c6a2df511f2eb5d30fd7e7fa))
- preserve optional context token cap in cli ([cd3e63c](https://github.com/NorthlandPositronics/Cogtrix/commit/cd3e63c2b02fcff9c5120e54973dd07fd1723ce8))
- protect \_bound_cache modifications in \_reset_for_new_run with lock (BUG-580) ([#620](https://github.com/NorthlandPositronics/Cogtrix/issues/620)) ([9b651c8](https://github.com/NorthlandPositronics/Cogtrix/commit/9b651c891f879d9007b82cb798c7b7d96bf767e1))
- protect \_tool_arg_schema_cache with threading.Lock in parallel execution ([#641](https://github.com/NorthlandPositronics/Cogtrix/issues/641)) ([d92d394](https://github.com/NorthlandPositronics/Cogtrix/commit/d92d394eea482fd0eb01670c743dee3567fd21ea))
- protect \_tool_call_counts with threading.Lock (issue [#667](https://github.com/NorthlandPositronics/Cogtrix/issues/667)) ([#673](https://github.com/NorthlandPositronics/Cogtrix/issues/673)) ([2ab0a15](https://github.com/NorthlandPositronics/Cogtrix/commit/2ab0a15629a2e751d1ec2b7e031a667bdcfb7909))
- **providers/openai:** allow OPENAI_API_KEY env-var fallback when base_url is set (closes [#991](https://github.com/NorthlandPositronics/Cogtrix/issues/991)) ([#1014](https://github.com/NorthlandPositronics/Cogtrix/issues/1014)) ([62b090a](https://github.com/NorthlandPositronics/Cogtrix/commit/62b090ab6636adcf291e5232b2eaa80c00ebe5af))
- **providers:** log debug message when \_extract_retry_after fails to parse response ([#1012](https://github.com/NorthlandPositronics/Cogtrix/issues/1012)) ([aaf5dd5](https://github.com/NorthlandPositronics/Cogtrix/commit/aaf5dd58c6c3f27644852896820cd236fc1210fe)), closes [#993](https://github.com/NorthlandPositronics/Cogtrix/issues/993)
- **providers:** raise ValueError when base_url is provided to Google provider ([4604495](https://github.com/NorthlandPositronics/Cogtrix/commit/4604495a5611c544b26b80b9064782ba173a954e))
- **providers:** redact base_url credentials in log messages (closes [#903](https://github.com/NorthlandPositronics/Cogtrix/issues/903)) ([#1054](https://github.com/NorthlandPositronics/Cogtrix/issues/1054)) ([e6cf233](https://github.com/NorthlandPositronics/Cogtrix/commit/e6cf2337a5874218890d0223587b413dc7b884ca))
- **providers:** use double-checked locking in \_load_provider to prevent race condition (issue [#807](https://github.com/NorthlandPositronics/Cogtrix/issues/807)) ([c5be3e8](https://github.com/NorthlandPositronics/Cogtrix/commit/c5be3e8737648cce7685708ed6aa33879c75c496))
- queue_after_tail considers past-due messages when computing queue tail (issue [#787](https://github.com/NorthlandPositronics/Cogtrix/issues/787)) ([503cfb4](https://github.com/NorthlandPositronics/Cogtrix/commit/503cfb42e1ae80ed621a99fa11fb7ad222512889))
- **rag:** replace unsafe FAISS pickle deserialization with safe raw format (closes [#1009](https://github.com/NorthlandPositronics/Cogtrix/issues/1009)) ([#1018](https://github.com/NorthlandPositronics/Cogtrix/issues/1018)) ([c702f04](https://github.com/NorthlandPositronics/Cogtrix/commit/c702f04f8468cae9298524cbe04c79f5aef7b9e2))
- raise RuntimeError when \_build_llm fails instead of returning None (issue [#794](https://github.com/NorthlandPositronics/Cogtrix/issues/794)) ([eec257b](https://github.com/NorthlandPositronics/Cogtrix/commit/eec257bc10195197b4d850ab52c5f0b54ff814bc))
- raise RuntimeError when \_build_memory_manager fails instead of returning None (issue [#795](https://github.com/NorthlandPositronics/Cogtrix/issues/795)) ([2bc1909](https://github.com/NorthlandPositronics/Cogtrix/commit/2bc19091e6bb61cc20b89a1220fc727162462598))
- record user message to memory on edit fallback send failure (issue [#680](https://github.com/NorthlandPositronics/Cogtrix/issues/680)) ([#690](https://github.com/NorthlandPositronics/Cogtrix/issues/690)) ([cb08e2e](https://github.com/NorthlandPositronics/Cogtrix/commit/cb08e2e015c60fc5fac3949d48dccdd0994abf7b))
- reinforce slack status dedup prompt ([55ec583](https://github.com/NorthlandPositronics/Cogtrix/commit/55ec5831f4cb2ba0960292c65d06c0b83514bfe6))
- **release-please:** post-process CHANGELOG.md to lint-clean output ([04afc90](https://github.com/NorthlandPositronics/Cogtrix/commit/04afc90fb0f8817610530983f65a2a32939bcf52))
- remove confirmation.py module-level singleton and wire configure() call (issue [#634](https://github.com/NorthlandPositronics/Cogtrix/issues/634)) ([#644](https://github.com/NorthlandPositronics/Cogtrix/issues/644)) ([f098231](https://github.com/NorthlandPositronics/Cogtrix/commit/f098231ab0326a4f54411e940a1fa508bdfdcf59))
- remove xfail from WebSocket resilience tests (issue [#741](https://github.com/NorthlandPositronics/Cogtrix/issues/741)) ([c64337f](https://github.com/NorthlandPositronics/Cogtrix/commit/c64337f884ea8fc7d9aadbdcb93f3a90288b0bb4))
- replace deprecated datetime.utcnow() with datetime.now(UTC) (issue [#796](https://github.com/NorthlandPositronics/Cogtrix/issues/796)) ([5c121c1](https://github.com/NorthlandPositronics/Cogtrix/commit/5c121c167c655a014d1499036034ce76123f8e30))
- resolve aiosqlite teardown collision for WebSocket tests (issue [#692](https://github.com/NorthlandPositronics/Cogtrix/issues/692)) ([#702](https://github.com/NorthlandPositronics/Cogtrix/issues/702)) ([a5f1b48](https://github.com/NorthlandPositronics/Cogtrix/commit/a5f1b48aa7ac8ab71a7be221ac2938183141cde3))
- return clear error when delegate agent returns empty response (issue [#811](https://github.com/NorthlandPositronics/Cogtrix/issues/811)) ([708f2f5](https://github.com/NorthlandPositronics/Cogtrix/commit/708f2f53b33b9f1bf9ccf31675fddba05b766746))
- sandbox delegate agents from destructive tools (issue [#713](https://github.com/NorthlandPositronics/Cogtrix/issues/713)) ([7e0176a](https://github.com/NorthlandPositronics/Cogtrix/commit/7e0176ae64c42439494e8849248aa4dc61e0270b))
- **scim:** support compound 'and' filters in SCIM user queries ([#124](https://github.com/NorthlandPositronics/Cogtrix/issues/124)) ([3eded69](https://github.com/NorthlandPositronics/Cogtrix/commit/3eded69ee36fbe38da7468fbb9eeefde481624e2))
- **scim:** support compound 'and' filters in SCIM user queries ([#124](https://github.com/NorthlandPositronics/Cogtrix/issues/124)) ([df03d74](https://github.com/NorthlandPositronics/Cogtrix/commit/df03d74d76fe189730254e045896cc52627abdb8))
- **scim:** validate PATCH paths and value types — RFC 7644 compliance ([#432](https://github.com/NorthlandPositronics/Cogtrix/issues/432)) ([d898451](https://github.com/NorthlandPositronics/Cogtrix/commit/d898451e2d553325e8b902001e01904cb4137882))
- **scim:** validate PATCH paths and value types — RFC 7644 compliance ([#432](https://github.com/NorthlandPositronics/Cogtrix/issues/432)) ([e273f0f](https://github.com/NorthlandPositronics/Cogtrix/commit/e273f0f4e161569af7c0fd7d2ce31c609fa0fb20))
- **security:** elevate 12 assistant endpoints from any-auth to admin-only (issue [#846](https://github.com/NorthlandPositronics/Cogtrix/issues/846)) ([#860](https://github.com/NorthlandPositronics/Cogtrix/issues/860)) ([e85521f](https://github.com/NorthlandPositronics/Cogtrix/commit/e85521f553b873372976480c469711f68c1ba1fd))
- sliding-window amnesia - cold-cache path truncates before summarization (issue [#657](https://github.com/NorthlandPositronics/Cogtrix/issues/657)) ([#660](https://github.com/NorthlandPositronics/Cogtrix/issues/660)) ([f2024f7](https://github.com/NorthlandPositronics/Cogtrix/commit/f2024f74d4fadc76c1f8eeaf0b3ebad76438abbe))
- stabilize tool call dedup keys for non-JSON args ([#498](https://github.com/NorthlandPositronics/Cogtrix/issues/498)) ([8f03e78](https://github.com/NorthlandPositronics/Cogtrix/commit/8f03e784f47b12abff159591fd1f84db9b6f176a))
- strengthen handler assertions and add UserCancelledRun path (issue [#691](https://github.com/NorthlandPositronics/Cogtrix/issues/691)) ([#697](https://github.com/NorthlandPositronics/Cogtrix/issues/697)) ([6a1a3f0](https://github.com/NorthlandPositronics/Cogtrix/commit/6a1a3f0957c5b65fd2465d2abc1f44afc7eeace0))
- strip CRLF and control chars from IMAP search fields to prevent protocol injection ([#876](https://github.com/NorthlandPositronics/Cogtrix/issues/876)) ([9a12305](https://github.com/NorthlandPositronics/Cogtrix/commit/9a12305789f1d471de74032ade1b0416b7021902)), closes [#870](https://github.com/NorthlandPositronics/Cogtrix/issues/870)
- suppress cron output in interactive console ([#171](https://github.com/NorthlandPositronics/Cogtrix/issues/171)) ([40fd584](https://github.com/NorthlandPositronics/Cogtrix/commit/40fd5846031fb20f23581ddb9baf1ffd8e1005bf))
- suppress cron output in interactive console ([#171](https://github.com/NorthlandPositronics/Cogtrix/issues/171)) ([bc38705](https://github.com/NorthlandPositronics/Cogtrix/commit/bc3870560f46c2f8a74d4b5390d0500b0260bd4c))
- **tests:** align test suite with post-bug-fix codebase ([#1253](https://github.com/NorthlandPositronics/Cogtrix/issues/1253)) ([b2b1df8](https://github.com/NorthlandPositronics/Cogtrix/commit/b2b1df8400dc272d9d3b50c2f64d3cfc804e1207))
- **tests:** avoid non-daemon thread blocking pytest shutdown in compression timeout test ([5fb4761](https://github.com/NorthlandPositronics/Cogtrix/commit/5fb47611b8f40381e8b6f9bac5e6465c9edcfa21))
- **tests:** avoid non-daemon thread blocking pytest shutdown in distill_summary timeout test ([d7a792f](https://github.com/NorthlandPositronics/Cogtrix/commit/d7a792ffcf8773cd0d5840aa34bf9f1e78a914fd))
- **tests:** avoid non-daemon thread blocking pytest shutdown in generate_tests timeout test ([d1d6b2e](https://github.com/NorthlandPositronics/Cogtrix/commit/d1d6b2e3ff708b3b86646f17ffeada9acb655d86))
- **tests:** avoid non-daemon thread blocking pytest shutdown in self_improve timeout tests ([d39c1cc](https://github.com/NorthlandPositronics/Cogtrix/commit/d39c1ccdb09c014f7af713f49c19b74e1d4b160e))
- **tests:** avoid non-daemon thread blocking pytest shutdown in threadpool timeout regression tests ([1a476a4](https://github.com/NorthlandPositronics/Cogtrix/commit/1a476a48501b55fc3bf1884ed0f7a415f334183c))
- **tests:** configure add_approval/revoke_approval mocks in test_self_improving_loop_features.py ([636edeb](https://github.com/NorthlandPositronics/Cogtrix/commit/636edeba9e4a180bd0004b27fa4f81d45e431fe4))
- **tests:** eliminate CI warning noise + aiosqlite lifecycle bug ([7b8ea53](https://github.com/NorthlandPositronics/Cogtrix/commit/7b8ea53047b729621439be26115a70059f2ac831))
- **tests:** import \_bootstrap from conftest to suppress import-time warnings ([d04579a](https://github.com/NorthlandPositronics/Cogtrix/commit/d04579ad05bcc51d449ea3e1ec2e236289c71e8f))
- **tests:** mark non-deterministic race test as xfail(strict=False) ([#1053](https://github.com/NorthlandPositronics/Cogtrix/issues/1053)) ([31e7c1f](https://github.com/NorthlandPositronics/Cogtrix/commit/31e7c1ffd7c0c602b5de754f37feeb739e808dfc))
- **tests:** move test_api_phase3 DB out of repo root into TMPDIR ([a8d9631](https://github.com/NorthlandPositronics/Cogtrix/commit/a8d9631dae9c72fc65815cd3b6bb283b80cf262e))
- **tests:** narrow except Exception to specific WS exception types (closes [#984](https://github.com/NorthlandPositronics/Cogtrix/issues/984)) ([#1046](https://github.com/NorthlandPositronics/Cogtrix/issues/1046)) ([15272f2](https://github.com/NorthlandPositronics/Cogtrix/commit/15272f2dba9207a195c55ba2f7d4f81051a3266c))
- **tests:** silence LangChainPendingDeprecationWarning during pytest ([f49d102](https://github.com/NorthlandPositronics/Cogtrix/commit/f49d1023e89ab61b9c7259b82f640a1bbf95f867))
- **tests:** skip TTY detection tests in CI runners ([#1235](https://github.com/NorthlandPositronics/Cogtrix/issues/1235)) ([5064a00](https://github.com/NorthlandPositronics/Cogtrix/commit/5064a005929174cd99a8f6a8431c2d005152657e)), closes [#1040](https://github.com/NorthlandPositronics/Cogtrix/issues/1040)
- **tests:** strengthen admin-bypass assertions in test_api_auth_security.py (closes [#982](https://github.com/NorthlandPositronics/Cogtrix/issues/982)) ([fcf0ec9](https://github.com/NorthlandPositronics/Cogtrix/commit/fcf0ec95babc7741a42e300ccb07dc9916f8e5ae))
- **tests:** update mocks for PR [#1208](https://github.com/NorthlandPositronics/Cogtrix/issues/1208) lock-protected approval methods ([f7519f5](https://github.com/NorthlandPositronics/Cogtrix/commit/f7519f5daf4817e0232149e10d36264244a615df))
- **tests:** use MagicMock for SQLAlchemy engine in health-readiness test ([532bab6](https://github.com/NorthlandPositronics/Cogtrix/commit/532bab6c13cab756cfabdfc6e63e3ab295fcda43))
- **tests:** use unique pgrep marker in orphan-cleanup shell test ([d07baab](https://github.com/NorthlandPositronics/Cogtrix/commit/d07baab072e19c8feb3092c3684ce52a1beee7cf))
- thread leak and cache eviction in orchestration ([#730](https://github.com/NorthlandPositronics/Cogtrix/issues/730), [#731](https://github.com/NorthlandPositronics/Cogtrix/issues/731), [#734](https://github.com/NorthlandPositronics/Cogtrix/issues/734)) ([5379aa4](https://github.com/NorthlandPositronics/Cogtrix/commit/5379aa4cc3e169962ddcf7c2b1965a121bfdd37a))
- token-budget working memory eviction ([9cfea92](https://github.com/NorthlandPositronics/Cogtrix/commit/9cfea927a24c3e5e87fae5a9b4685d757280dfef))
- **tooling:** preserve duplicate detection for non-serializable args ([68948f4](https://github.com/NorthlandPositronics/Cogtrix/commit/68948f48bc5dce1384c2495a18c172d2bc2c620a))
- **tools/self_improve:** check ruff/bandit return codes and surface warnings on failure (closes [#969](https://github.com/NorthlandPositronics/Cogtrix/issues/969)) ([#1015](https://github.com/NorthlandPositronics/Cogtrix/issues/1015)) ([cd29e1c](https://github.com/NorthlandPositronics/Cogtrix/commit/cd29e1cea9a23e501298be0293efde13d10552e0))
- **tools/shell:** handle os.killpg ESRCH to prevent orphaned grandchild processes (closes [#966](https://github.com/NorthlandPositronics/Cogtrix/issues/966)) ([#1052](https://github.com/NorthlandPositronics/Cogtrix/issues/1052)) ([b7f71c1](https://github.com/NorthlandPositronics/Cogtrix/commit/b7f71c1fab2c6c54bd773e458f13aee0d95463ba))
- **tools/shell:** remove { and } from shell metachar set ([#1008](https://github.com/NorthlandPositronics/Cogtrix/issues/1008)) ([2b54827](https://github.com/NorthlandPositronics/Cogtrix/commit/2b5482759e49a8d7a713aae9f72cca51f0d18cfb)), closes [#943](https://github.com/NorthlandPositronics/Cogtrix/issues/943)
- **tools:** add COGTRIX_ENABLE_DATASCIENCE_MODULES config flag to close numpy sandbox escape ([9e75da9](https://github.com/NorthlandPositronics/Cogtrix/commit/9e75da904fac139b6b387b461549d8516d2d3397))
- **tools:** add COGTRIX_ENABLE_DATASCIENCE_MODULES config flag to close numpy sandbox escape ([87db24a](https://github.com/NorthlandPositronics/Cogtrix/commit/87db24a7bab878bf7b6783fd89cfc42987bd80f3))
- **tools:** add COGTRIX_ENABLE_DATASCIENCE_MODULES config flag to close numpy sandbox escape ([#1265](https://github.com/NorthlandPositronics/Cogtrix/issues/1265)) ([9e75da9](https://github.com/NorthlandPositronics/Cogtrix/commit/9e75da904fac139b6b387b461549d8516d2d3397))
- **tools:** add dict-path unwrapping to write_file, append_file, patch_file (closes [#1028](https://github.com/NorthlandPositronics/Cogtrix/issues/1028)) ([fd67160](https://github.com/NorthlandPositronics/Cogtrix/commit/fd671601116e937ad1680475f775fceebd4081ce))
- **tools:** add file locking to patch_file read-modify-write cycle ([#1250](https://github.com/NorthlandPositronics/Cogtrix/issues/1250)) ([7ef8683](https://github.com/NorthlandPositronics/Cogtrix/commit/7ef8683dce37efd4616a4a83f064ccabb2ac747e))
- **tools:** add missing shell metacharacters ~ \ ! # to \_shell_meta (closes [#1074](https://github.com/NorthlandPositronics/Cogtrix/issues/1074)) ([151c177](https://github.com/NorthlandPositronics/Cogtrix/commit/151c1774903a4268220517247dd7dfedf37202c0))
- **tools:** add pattern/expression → schedule remaps for cron_add ([#520](https://github.com/NorthlandPositronics/Cogtrix/issues/520), [#523](https://github.com/NorthlandPositronics/Cogtrix/issues/523)) ([#530](https://github.com/NorthlandPositronics/Cogtrix/issues/530)) ([81cdf82](https://github.com/NorthlandPositronics/Cogtrix/commit/81cdf82699045b9286273a9e994d8d8a47c6e03e))
- **tools:** add ThreadPoolExecutor timeout to generate_tests llm.invoke() ([0704810](https://github.com/NorthlandPositronics/Cogtrix/commit/070481021ca48f329a79da6a2adf11fd373a2dcc))
- **tools:** add ThreadPoolExecutor timeout to generate_tests llm.invoke() ([fa590f0](https://github.com/NorthlandPositronics/Cogtrix/commit/fa590f07bf3d01ac533839352b7a77842c00f80b))
- **tools:** add ThreadPoolExecutor timeout to self_improve llm.invoke() ([5368df2](https://github.com/NorthlandPositronics/Cogtrix/commit/5368df25e3ddb4b936617c4fe47b25413a5db1ab))
- **tools:** add ThreadPoolExecutor timeout to self_improve llm.invoke() ([0dc6877](https://github.com/NorthlandPositronics/Cogtrix/commit/0dc6877bc6177b35ed4f01c036fba9fc86ade65a))
- **tools:** add working_directory boundary validation to execute_shell_command (closes [#1029](https://github.com/NorthlandPositronics/Cogtrix/issues/1029)) ([#1068](https://github.com/NorthlandPositronics/Cogtrix/issues/1068)) ([913aa4e](https://github.com/NorthlandPositronics/Cogtrix/commit/913aa4e58bfe4c3a24b3d9cc596e7d5e7d88b6a9))
- **tools:** block command substitution $() and backticks in shell tool ([16ad532](https://github.com/NorthlandPositronics/Cogtrix/commit/16ad53224f8f3160e2fbfcfd95a3b0a75932019b))
- **tools:** block command substitution $() and backticks in shell tool ([8efa23a](https://github.com/NorthlandPositronics/Cogtrix/commit/8efa23ac79252ed9e0f0180a031e219da5711ec5)), closes [#1104](https://github.com/NorthlandPositronics/Cogtrix/issues/1104)
- **tools:** block command substitution $() and backticks in shell tool ([#1107](https://github.com/NorthlandPositronics/Cogtrix/issues/1107)) ([16ad532](https://github.com/NorthlandPositronics/Cogtrix/commit/16ad53224f8f3160e2fbfcfd95a3b0a75932019b))
- **tools:** delegate UserCancelledRun propagation in handler and agent ([cfc87bc](https://github.com/NorthlandPositronics/Cogtrix/commit/cfc87bc22409bff2aa468abcbaa28c6244e12bd4))
- **tools:** delegate UserCancelledRun propagation in handler and agent ([dbf590e](https://github.com/NorthlandPositronics/Cogtrix/commit/dbf590eb54681283428dbe8bbb42531750b78500))
- **tools:** explicitly reject dangling symlinks in \_validate_path ([#996](https://github.com/NorthlandPositronics/Cogtrix/issues/996)) ([c5fef8c](https://github.com/NorthlandPositronics/Cogtrix/commit/c5fef8cd8c7cc66e02766ffb0b2fefb888670431)), closes [#945](https://github.com/NorthlandPositronics/Cogtrix/issues/945)
- **tools:** isolate python_exec default sessions per execution context (closes [#1073](https://github.com/NorthlandPositronics/Cogtrix/issues/1073)) ([8c0f0b7](https://github.com/NorthlandPositronics/Cogtrix/commit/8c0f0b75b52fadc1763b81daa7628cb6da4cf336))
- **tools:** log warning when loop limiter AST transformation fails ([#1153](https://github.com/NorthlandPositronics/Cogtrix/issues/1153)) ([e69f3f8](https://github.com/NorthlandPositronics/Cogtrix/commit/e69f3f869ce3ae5cdcdedb37e5f03afcb068250e))
- **tools:** prevent list_directory from following symlinks in glob (closes [#944](https://github.com/NorthlandPositronics/Cogtrix/issues/944)) ([#1002](https://github.com/NorthlandPositronics/Cogtrix/issues/1002)) ([a96d121](https://github.com/NorthlandPositronics/Cogtrix/commit/a96d121358cb378bd4206626077f3b1ccb347c93))
- **tools:** prevent list_directory from following symlinks in glob (closes [#944](https://github.com/NorthlandPositronics/Cogtrix/issues/944)) ([#1055](https://github.com/NorthlandPositronics/Cogtrix/issues/1055)) ([9b93d2c](https://github.com/NorthlandPositronics/Cogtrix/commit/9b93d2c70efba98eb334a1ab2a6b6b56687652c6))
- **tools:** reject code when loop limiter AST transformation fails ([#1248](https://github.com/NorthlandPositronics/Cogtrix/issues/1248)) ([4c7d345](https://github.com/NorthlandPositronics/Cogtrix/commit/4c7d3456de4710b75333af1dffc1f095a9ffce03)), closes [#1240](https://github.com/NorthlandPositronics/Cogtrix/issues/1240)
- unify session ID sanitization via shared path_safety module (issue [#670](https://github.com/NorthlandPositronics/Cogtrix/issues/670)) ([#685](https://github.com/NorthlandPositronics/Cogtrix/issues/685)) ([e326aec](https://github.com/NorthlandPositronics/Cogtrix/commit/e326aecbddd95ed0909c405961164f3d66ebd16d))
- unify session-ID sanitization by importing from shared path_safety module (issue [#715](https://github.com/NorthlandPositronics/Cogtrix/issues/715)) ([a56e805](https://github.com/NorthlandPositronics/Cogtrix/commit/a56e80500fb14ac63350ce835b542ef7e75320f8))
- update delegate tool tests to assert sandboxed behavior (issue [#784](https://github.com/NorthlandPositronics/Cogtrix/issues/784)) ([8aff1fb](https://github.com/NorthlandPositronics/Cogtrix/commit/8aff1fb79c6cb28b6b4799658f3af2edee2adcb1))
- use prerecord_user() for shutdown durability without breaking deferral tests (issue [#681](https://github.com/NorthlandPositronics/Cogtrix/issues/681)) ([0fb0da9](https://github.com/NorthlandPositronics/Cogtrix/commit/0fb0da9545d3a3d9c29b7ae658410ebcecea83c0))
- use urlparse hostname match for DeepSeek detection (issue [#669](https://github.com/NorthlandPositronics/Cogtrix/issues/669)) ([#682](https://github.com/NorthlandPositronics/Cogtrix/issues/682)) ([e9d17ce](https://github.com/NorthlandPositronics/Cogtrix/commit/e9d17cef27d5ac8b0dd46606358a8f1705006346))
- verify API key mask format first-3+\*\*\*+last-4 (issue [#749](https://github.com/NorthlandPositronics/Cogtrix/issues/749)) ([34ad6d3](https://github.com/NorthlandPositronics/Cogtrix/commit/34ad6d383a530915cca04eecce156f8148cce496))
- wire build_process_tools_node() into build_agent_graph() (issue [#630](https://github.com/NorthlandPositronics/Cogtrix/issues/630)) ([#647](https://github.com/NorthlandPositronics/Cogtrix/issues/647)) ([e95cc09](https://github.com/NorthlandPositronics/Cogtrix/commit/e95cc09542bc45f37c7553b517ebb36694962c55))
- wire filter_tools_for_agent into CLI task runner for agent tool restrictions (issue [#684](https://github.com/NorthlandPositronics/Cogtrix/issues/684)) ([#706](https://github.com/NorthlandPositronics/Cogtrix/issues/706)) ([53ebc5d](https://github.com/NorthlandPositronics/Cogtrix/commit/53ebc5d0c3eab1174f873ed0397d6f1b4312a909))

### Documentation

- add ADR for build_agent_graph node extraction ([#549](https://github.com/NorthlandPositronics/Cogtrix/issues/549)) ([2f6a94a](https://github.com/NorthlandPositronics/Cogtrix/commit/2f6a94ad89db2e529d8a01ddd4b60f211bef9dc3))
- add K8s deployment topology spec (issue [#600](https://github.com/NorthlandPositronics/Cogtrix/issues/600)) ([#659](https://github.com/NorthlandPositronics/Cogtrix/issues/659)) ([5defa08](https://github.com/NorthlandPositronics/Cogtrix/commit/5defa08133b2477f8547dac822940e3c2c8c8dc0))
- **adr:** commit ADR-0054 (Enterprise SSO) and ADR-0055 (RBAC) ([#1205](https://github.com/NorthlandPositronics/Cogtrix/issues/1205)) ([1f21bb6](https://github.com/NorthlandPositronics/Cogtrix/commit/1f21bb6f60beb7bf7866f62e27c18593a7c5c254))
- **api:** document 14 enterprise API route groups ([#437](https://github.com/NorthlandPositronics/Cogtrix/issues/437)) ([ed9cce2](https://github.com/NorthlandPositronics/Cogtrix/commit/ed9cce2e88eed709029b5dd2d4c793d70ecc3daf))
- **api:** document 14 enterprise API route groups ([#437](https://github.com/NorthlandPositronics/Cogtrix/issues/437)) ([593c548](https://github.com/NorthlandPositronics/Cogtrix/commit/593c54807fae77e40da47eaf1d19f7ecf42ad8a8))
- **api:** fix WS idle timeout (90s → 300s) and document DonePayload text field ([fc0127d](https://github.com/NorthlandPositronics/Cogtrix/commit/fc0127d0072b0934f073232eca56d882b0ad38f2)), closes [#440](https://github.com/NorthlandPositronics/Cogtrix/issues/440)
- **api:** fix WS idle timeout (90s → 300s) and document DonePayload text field ([#440](https://github.com/NorthlandPositronics/Cogtrix/issues/440)) ([c14ceaa](https://github.com/NorthlandPositronics/Cogtrix/commit/c14ceaa86f7d0e88718507486ef0db155d0a1bce))
- **api:** sync client-contract.md with source schema drift ([#439](https://github.com/NorthlandPositronics/Cogtrix/issues/439)) ([575ea1e](https://github.com/NorthlandPositronics/Cogtrix/commit/575ea1ea947ffef993389fb20ca7a57a8c014e2e))
- **api:** sync client-contract.md with source schema drift ([#439](https://github.com/NorthlandPositronics/Cogtrix/issues/439)) ([f25c50f](https://github.com/NorthlandPositronics/Cogtrix/commit/f25c50fdd8163a91f57f44190106d80884c4aacd))
- consolidate markdownlint config and fix all 170 pre-existing violations ([1a4a2e2](https://github.com/NorthlandPositronics/Cogtrix/commit/1a4a2e2a964b3d93269b4c9bee74e55373924b28))
- **contributing:** wrap canary policy pointer to satisfy MD013/300 ([52efcfa](https://github.com/NorthlandPositronics/Cogtrix/commit/52efcfa51f3892d3a8e6e43d451aef3073012be3))
- fix ARCHITECTURE.md missing files, SearXNG env var, CHANGELOG duplicate ([#441](https://github.com/NorthlandPositronics/Cogtrix/issues/441)) ([c9a5420](https://github.com/NorthlandPositronics/Cogtrix/commit/c9a54201262c43388930b17f4a8db56430dc13b0))
- fix ARCHITECTURE.md missing files, SearXNG env var, CHANGELOG duplicate ([#441](https://github.com/NorthlandPositronics/Cogtrix/issues/441)) ([fc03b47](https://github.com/NorthlandPositronics/Cogtrix/commit/fc03b47c04e7d725b1cfc8c9c648eeeca45ff1f1))
- fix INDEX and DEVELOPMENT doc gaps ([#442](https://github.com/NorthlandPositronics/Cogtrix/issues/442)) ([9452c68](https://github.com/NorthlandPositronics/Cogtrix/commit/9452c683f793534fb4c8e2c4d46b369673f6dd72))
- fix INDEX.md + DEVELOPMENT.md gaps ([#442](https://github.com/NorthlandPositronics/Cogtrix/issues/442)) ([c355f9e](https://github.com/NorthlandPositronics/Cogtrix/commit/c355f9e541bdf59692e166beb1efad0781ad4220))
- fix INDEX.md + DEVELOPMENT.md gaps ([#442](https://github.com/NorthlandPositronics/Cogtrix/issues/442)) ([a68d938](https://github.com/NorthlandPositronics/Cogtrix/commit/a68d93888a7479cecd8f887673fc10722c8e35cc))
- fix tool count and add slack_tools.py documentation ([#698](https://github.com/NorthlandPositronics/Cogtrix/issues/698), [#699](https://github.com/NorthlandPositronics/Cogtrix/issues/699)) ([#705](https://github.com/NorthlandPositronics/Cogtrix/issues/705)) ([2ab444a](https://github.com/NorthlandPositronics/Cogtrix/commit/2ab444ac903eaded7dbcb6dd8bd41ec6b7af9e23))
- fix version numbers across README, ROADMAP, VERSIONING, pyproject.toml ([f997d3a](https://github.com/NorthlandPositronics/Cogtrix/commit/f997d3ac94bcc01656c0900544286ef09d5a2573))
- fix version numbers across README, ROADMAP, VERSIONING, pyproject.toml ([aa6e487](https://github.com/NorthlandPositronics/Cogtrix/commit/aa6e487a1ad6830f7deda6e11b8207ee64a0432a)), closes [#438](https://github.com/NorthlandPositronics/Cogtrix/issues/438)
- **quality:** add QUALITY_PIPELINE.md — holistic two-gate + session monitoring design ([#496](https://github.com/NorthlandPositronics/Cogtrix/issues/496)) ([2a38e93](https://github.com/NorthlandPositronics/Cogtrix/commit/2a38e930d27ff4f22e82481f7e5ad8a4822d5560))
- update index and development references ([d4d24f4](https://github.com/NorthlandPositronics/Cogtrix/commit/d4d24f409822abc710561a1cdbe004645bfa7387))

## [0.2.6](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.5...v0.2.6) (2026-04-26)

### Bug Fixes

- **docker:** harden Dockerfiles — /data volume, read-only /app, security fixes ([#52](https://github.com/NorthlandPositronics/Cogtrix/issues/52)) ([e9073f8](https://github.com/NorthlandPositronics/Cogtrix/commit/e9073f8419a2e53e8fe13fafa643c2a88125422b))

## [0.2.5](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.4...v0.2.5) (2026-04-25)

### Bug Fixes

- sync uv.lock and regenerate requirements.txt ([#30](https://github.com/NorthlandPositronics/Cogtrix/issues/30)) ([#31](https://github.com/NorthlandPositronics/Cogtrix/issues/31)) ([0f86c1a](https://github.com/NorthlandPositronics/Cogtrix/commit/0f86c1ad4357300d60e2d177f3b0a26809b402a4))

## [Unreleased]

### Features

- **decision accountability** — opt-in self-debate framework for autonomous execution (ADR-0052). When `decision_accountability.enabled: true`, the agent generates a structured plan with assumptions and evidence, produces a counter-plan, and appends an uncertainty note when adjusted confidence falls below the threshold. Adds `ACCOUNTABILITY_PROMPT` injection to system prompt and post-response
  parsing in `graph.py`. Off by default; no breaking changes.

### Documentation

- `docs/CONFIGURATION.md` — new Decision Accountability section with options table, how-it-works, migration note
- `docs/adr/0052-decision-accountability.md` — status updated to Accepted; all milestones marked complete

---

## [0.2.4](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.3...v0.2.4) (2026-04-06)

### Features

- agent complexity awareness — checkpoints, stuck detection, work cycle, metrics ([e6a700c](https://github.com/NorthlandPositronics/Cogtrix/commit/e6a700c9dc0cb9a895e8709deed1b25020fb7572))
- agent complexity awareness — checkpoints, stuck detection, work cycle, metrics ([#326](https://github.com/NorthlandPositronics/Cogtrix/issues/326)) ([e6a700c](https://github.com/NorthlandPositronics/Cogtrix/commit/e6a700c9dc0cb9a895e8709deed1b25020fb7572))
- checkpoint tool and periodic reflection for complex tasks ([badb9c8](https://github.com/NorthlandPositronics/Cogtrix/commit/badb9c83413a23445d26a28e4b26b2961061e780))
- checkpoint-based stuck detection — force thinking break after 15 rounds without progress ([0e4fa06](https://github.com/NorthlandPositronics/Cogtrix/commit/0e4fa06388754792201fc15065e724bc29b52846))
- configurable per-model LLM request timeout ([ecefd18](https://github.com/NorthlandPositronics/Cogtrix/commit/ecefd18f81eb17b3e95b2226fcb1e19ee73bd60e))
- context-aware periodic reflection — debug mode vs work cycle ([269f84f](https://github.com/NorthlandPositronics/Cogtrix/commit/269f84f43803102fd98c51e1308bef2ee46921da))
- debugging guidance — isolate, diagnose, search, change one thing ([7959f71](https://github.com/NorthlandPositronics/Cogtrix/commit/7959f71165c5d397a53c00ea78d4633fcc117b64))
- scale stuck threshold, checkpoint nudge, rewrite detection ([ac4566c](https://github.com/NorthlandPositronics/Cogtrix/commit/ac4566c1af68c82d2b3f9862179742b5f37e4ce8))
- structured work cycle (RESEARCH→ANALYZE→ACT→EVALUATE→PIVOT) ([8c45950](https://github.com/NorthlandPositronics/Cogtrix/commit/8c45950c5a04ea3bfa27889e715c700f06937906))
- stuck detection with forced thinking break ([5712d56](https://github.com/NorthlandPositronics/Cogtrix/commit/5712d56ab824060f2d07055882cab8c8832e72f4))
- task complexity awareness with adaptive execution strategy ([1b034e3](https://github.com/NorthlandPositronics/Cogtrix/commit/1b034e3f09537e5fcbc94b2785d83f14457a13fb))
- **ui:** redesign tool confirmation panel per UX spec ([b3a843f](https://github.com/NorthlandPositronics/Cogtrix/commit/b3a843f902b5afb775f4f72f72c5089443b579b7))

### Bug Fixes

- add DEFAULT_CONTEXT_WINDOW = 32768 to ModelConfig ([d6a882c](https://github.com/NorthlandPositronics/Cogtrix/commit/d6a882c9d7c9c99defcadb5fc4ef2a1749b1d118))
- add INFO logging for checkpoint nudge, confirm it was firing ([3f68aee](https://github.com/NorthlandPositronics/Cogtrix/commit/3f68aeefb6a5aee1726009512bd78f9b1c82571a))
- address code quality review — remove dead code, document empty excepts ([fea7a0b](https://github.com/NorthlandPositronics/Cogtrix/commit/fea7a0b80032c7d6bd2bb9fd9a2704708976613a))
- address code quality review from PR [#327](https://github.com/NorthlandPositronics/Cogtrix/issues/327) ([983ac95](https://github.com/NorthlandPositronics/Cogtrix/commit/983ac95123ef60aa578d67fec322b341268cf6d3))
- address code quality review from PR [#327](https://github.com/NorthlandPositronics/Cogtrix/issues/327) ([#328](https://github.com/NorthlandPositronics/Cogtrix/issues/328)) ([983ac95](https://github.com/NorthlandPositronics/Cogtrix/commit/983ac95123ef60aa578d67fec322b341268cf6d3))
- auto-load search_web, heredoc rewrite detection, debug isolation ([0322f8d](https://github.com/NorthlandPositronics/Cogtrix/commit/0322f8db873f9d0cc84be3d43c7dbcca6dc649bd))
- broaden stuck detection error indicators and add success check ([5c09a99](https://github.com/NorthlandPositronics/Cogtrix/commit/5c09a99406900e9158020d1f377a1502751d9f58))
- budget-disabled tools added to denials; accept command/file_path aliases ([cc868a4](https://github.com/NorthlandPositronics/Cogtrix/commit/cc868a43a00cfd06b7741dc41e76527ac16d8a55))
- checkpoint nudge ordering, debug oscillation guidance ([933eea3](https://github.com/NorthlandPositronics/Cogtrix/commit/933eea3fbf1f3188dda44d648af5b60cb11f4b9a))
- confirmation panel — bright underlined choices, action description ([ae459ed](https://github.com/NorthlandPositronics/Cogtrix/commit/ae459ed0def1026d911aa8837e82b47aa9b5e5b1))
- CONFIRMED ABSENT checkpoint pattern, raise MODERATE threshold 15→20 ([671ed9f](https://github.com/NorthlandPositronics/Cogtrix/commit/671ed9f65789b1ccf42caa2bf88754e4e5ebbbc1))
- debug reflection scans recent messages instead of consecutive counter ([48a265b](https://github.com/NorthlandPositronics/Cogtrix/commit/48a265b77d41d65d0682e890634abc43d9cc4504))
- exempt action tools from budget, improve denied-tool messaging ([0da8c3f](https://github.com/NorthlandPositronics/Cogtrix/commit/0da8c3f92f824efb9469d2e29ccdaf692f815312))
- guard None temperature in delegate_parallel ([9bf0094](https://github.com/NorthlandPositronics/Cogtrix/commit/9bf0094071a8cedb79327df37329b5bb95b9939f))
- guard None temperature in delegate_parallel to prevent TypeError ([94972b4](https://github.com/NorthlandPositronics/Cogtrix/commit/94972b4213176e155b88610bc98cff6f09b3038b))
- guard None temperature in delegate_parallel, unify defaults to 0.5 ([#325](https://github.com/NorthlandPositronics/Cogtrix/issues/325)) ([9bf0094](https://github.com/NorthlandPositronics/Cogtrix/commit/9bf0094071a8cedb79327df37329b5bb95b9939f))
- LLM call timeout prevents indefinite hangs on backend disconnect ([dd36dc5](https://github.com/NorthlandPositronics/Cogtrix/commit/dd36dc50323ca8aadc669f456e2c850249cfa059))
- lower reflection interval to 10, checkpoint both successes and failures ([726741a](https://github.com/NorthlandPositronics/Cogtrix/commit/726741a7c033d13c4e84e7cac3ba5737a35e8e26))
- move ModelConfig import to top of file (ruff E402) ([ed42602](https://github.com/NorthlandPositronics/Cogtrix/commit/ed4260220d9bb25d1dfc7c3e8b4c99d9d51e1ce8))
- raise COMPLEX_ACTION stuck threshold, checkpoint exact commands ([2e24958](https://github.com/NorthlandPositronics/Cogtrix/commit/2e24958b9e3a9d685f7ec0f3010a5e53399a072a))
- RBA forcing, pivot quality, rewrite threshold — from metrics analysis ([9cbd235](https://github.com/NorthlandPositronics/Cogtrix/commit/9cbd235aa3b9474a8a86aacbbd5f3c95b6131966))
- remove extend_run from always-active tools — caused tool confusion ([d7f997d](https://github.com/NorthlandPositronics/Cogtrix/commit/d7f997d5e493607b2629da335da3c0c57e7d23ca))
- remove square brackets from confirmation choices, fix logging indent ([3ee7164](https://github.com/NorthlandPositronics/Cogtrix/commit/3ee71648219227b9f157212331b309af85c23baa))
- remove test-case-specific examples from system prompt and tool descriptions ([7988522](https://github.com/NorthlandPositronics/Cogtrix/commit/798852204e3f1cba801e1aa3a761c25439d5434e))
- RESEARCH step must use web search, not guess URLs ([237b1b8](https://github.com/NorthlandPositronics/Cogtrix/commit/237b1b8b92820c82a731ccb724bc35c3cb441d53))
- resolve LazyToolProxy before auto-loading search_web ([ded35f7](https://github.com/NorthlandPositronics/Cogtrix/commit/ded35f7ddf6e0d8bd8cc33cb4fdb0f682a6542aa))
- shell tool confirmation panel showed "(no parameters)" ([19054f3](https://github.com/NorthlandPositronics/Cogtrix/commit/19054f3f1c652346b012b3a5bb2f6f9afe215599))
- skip forced delegation when model already produced substantial response ([0886668](https://github.com/NorthlandPositronics/Cogtrix/commit/0886668a67f4f8084bad7d7e9ae9f4a51b6e3fa8))
- skip forced delegation when model already produced substantial response ([429e6eb](https://github.com/NorthlandPositronics/Cogtrix/commit/429e6ebbe2c165916599349c6ed6b98afbfbaa8e))
- system prompt efficiency guidance — batch scripts, no re-checking, read errors ([56f5ae0](https://github.com/NorthlandPositronics/Cogtrix/commit/56f5ae0460bccd5e2bef916e3bd3644b7f7f3ccf))
- tighten COMPLEX_RESEARCH classifier — remove enumeration pattern ([edb1993](https://github.com/NorthlandPositronics/Cogtrix/commit/edb19932a1a4a04aee27e02dfc7c8604329382ba))
- **tools:** shell tool description — guide model to set correct timeout ([3bde70e](https://github.com/NorthlandPositronics/Cogtrix/commit/3bde70e5606d2592627bd6ca555920f14ba663ec))
- unify default temperature to 0.5 across all code paths ([3af0bfc](https://github.com/NorthlandPositronics/Cogtrix/commit/3af0bfc5fd584ec774eb5923353c19cb34d9bc2d))
- **ux:** hide tool_call metadata from confirmation panel; remove prompt indent ([e1ecdf9](https://github.com/NorthlandPositronics/Cogtrix/commit/e1ecdf9831dc40546e014009b465d0096e44c655))
- **ux:** move choices inside panel, inline highlighted key letters ([9f75ba8](https://github.com/NorthlandPositronics/Cogtrix/commit/9f75ba812f5b4c9154794f57cd53585b4be975d0))
- **ux:** remove leading spaces from 'Logging to:' line ([61353c0](https://github.com/NorthlandPositronics/Cogtrix/commit/61353c0b2b43525adcff2f1cedd28f3b55f5730b))
- **ux:** unwrap LangChain tool_call envelope in confirmation panel ([2d4c710](https://github.com/NorthlandPositronics/Cogtrix/commit/2d4c7109d5cea500103fbcbe0e3707418ff4b7f8))
- work cycle ACT step reminds agent to load execution tools ([7556627](https://github.com/NorthlandPositronics/Cogtrix/commit/7556627f9d23dd396c6aa8c5a7e4b2563f6a469d))
- write_file /tmp access, parameter remaps, CoT research cycle ([72471c8](https://github.com/NorthlandPositronics/Cogtrix/commit/72471c8a173c9145f44f57ec20ac429c67e773c2))

### Documentation

- 10 agent effectiveness metrics with baselines and targets ([b0b0607](https://github.com/NorthlandPositronics/Cogtrix/commit/b0b0607b62d19636ccac02f51cf6e81d886b1909))
- comprehensive test run documentation for agent complexity work ([633c271](https://github.com/NorthlandPositronics/Cogtrix/commit/633c271aad3684f661975047f99d4d522446b0ff))
- document timeout architecture in CLAUDE.md ([cfb1cc4](https://github.com/NorthlandPositronics/Cogtrix/commit/cfb1cc46249d2d7d6555c6fd5ca633fdd7eb98f0))
- parallel run 4 metrics scorecard — all 10 metrics applied ([711fbe4](https://github.com/NorthlandPositronics/Cogtrix/commit/711fbe45bd528e7525b216cc038ad0a03ecec28f))
- parallel run 5 metrics — composite 64→71, 7/10 metrics at target ([063a8f6](https://github.com/NorthlandPositronics/Cogtrix/commit/063a8f653033a1faae19356bbd8c298d0854b5c0))
- parallel run 6 metrics — RBA 53→74%, 8/10 at target ([3c34a06](https://github.com/NorthlandPositronics/Cogtrix/commit/3c34a0603d27294db01f96e4e9e288d58cc32183))
- parallel run 7 — composite 77, TCR 85%, 4/5 complete (best ever) ([1b7352a](https://github.com/NorthlandPositronics/Cogtrix/commit/1b7352a55ac5eb71b79584cc7a3c801c0dc5ccea))
- parallel run 8 — composite 80, 9/10 metrics at target, DLE passes ([4025c49](https://github.com/NorthlandPositronics/Cogtrix/commit/4025c490859908e0a32a759caf9f8612fe4a5520))
- update CLAUDE.md with task complexity, checkpoint, stuck detection ([7fd30f0](https://github.com/NorthlandPositronics/Cogtrix/commit/7fd30f04d4f6ae3975c7a27b58cc03ec01fd40b2))

## [0.2.3](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.2...v0.2.3) (2026-04-03)

### Bug Fixes

- **critical:** extract_response returned stale answer from previous turn ([d65a0f8](https://github.com/NorthlandPositronics/Cogtrix/commit/d65a0f875f31ce4f2a4ccad6fb8caf758bb1b65a))

## [0.2.2](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.2.1...v0.2.2) (2026-04-03)

### Features

- TTFT optimizations, tool budget, forge audits, Ollama token fix ([61a992c](https://github.com/NorthlandPositronics/Cogtrix/commit/61a992c0c666649cba8500c04551a2cf6e41f921))

### Bug Fixes

- restore release-please manifest to 0.2.1 ([b0ba369](https://github.com/NorthlandPositronics/Cogtrix/commit/b0ba3694b67d2f217933c299816606dc1cd2453b))

## [0.2.0] — 2026-03-31

### Highlights

This release marks Cogtrix's first production milestone. It consolidates all backend infrastructure (M5–M7) and the UX sprint series (Sprints 1–4) into a stable, versioned public release.

### Infrastructure (M5–M7)

- **M5**: PostgreSQL as default DB, Redis session presence, OIDC/SSO integration, structured audit log, per-user resource quotas
- **M6**: CogTrixGo feature parity — 60 tools, Anthropic/Google providers, multi-agent orchestration, REST API, WhatsApp/Telegram assistant mode
- **M7**: Security audit (bandit), load test (p95 = 0.271 s), API stability guarantee (VERSIONING.md + CI breaking-change check), migration guide, Docker multi-arch image (bake.hcl)

### UX Sprints (Sprints 1–4)

- **Sprint 1** (#215–#217): COGTRIX.md auto-load, memory visibility (/memory), @file/@folder inline context injection
- **Sprint 2** (#218–#220): diff view before writes, tool name in spinner + cumulative token counter, auto model routing (--auto-route/-R)
- **Sprint 3** (#221–#224): --quick/-Q fast path, session export/share (/export), context advisor (≥70% warning / ≥85% critical), git-native mode (--git-native/-G)
- **Sprint 4** (#225–#228): shell completion (--install-completion), compact banner (banner: compact/off, --no-banner), per-tool trust (tool_trust config), named profiles (--profile/-P)

### Bug Fixes

See individual milestone changelogs for the full list of bug fixes (BUG-001 through BUG-236).

## [0.1.38](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.37...v0.1.38) (2026-03-26)

### Bug Fixes

- **api:** BUG-252/253 — deep think mode answer disappears ([#103](https://github.com/NorthlandPositronics/Cogtrix/issues/103)) ([f4803aa](https://github.com/NorthlandPositronics/Cogtrix/commit/f4803aaf08c1f1c1ed4b2faf9bec5cd9e1b9350c))

## [0.1.37](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.36...v0.1.37) (2026-03-25)

### Bug Fixes

- **api:** BUG-248/249/250/251 — API think mode always runs force_deep_think; delegate tools wired in worker threads ([fcf5394](https://github.com/NorthlandPositronics/Cogtrix/commit/fcf539422cfc398e21bf7d313ba80c730f049dec))

## [0.1.36](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.35...v0.1.36) (2026-03-25)

### Features

- **api:** permanent delete + restore sessions; runtime provider CRUD ([4630c77](https://github.com/NorthlandPositronics/Cogtrix/commit/4630c77e12391c68fe12daba651c78ec660cfaf9))
- **api:** permanent session delete/restore + runtime provider CRUD ([17b419c](https://github.com/NorthlandPositronics/Cogtrix/commit/17b419c62e085fa5c724e8686001c6b08e083881))
- **cli:** implement CLI output style guide across all slash commands ([e7f3dd2](https://github.com/NorthlandPositronics/Cogtrix/commit/e7f3dd2f932ad804d72e488ba4940d9f1e53b128))
- **scripts:** add ProjectForge prompt suite and cogtrix-task Docker runner ([b61c709](https://github.com/NorthlandPositronics/Cogtrix/commit/b61c709b0898f5e8d052ade4310a0f5a526c8c3d))
- self-improving loop foundation — 7 targeted improvements ([e16b9dd](https://github.com/NorthlandPositronics/Cogtrix/commit/e16b9dd7fee276dcdffc0e2ba9488a96fafcd1e6))
- self-improving loop, API improvements, and bug fixes ([23e593b](https://github.com/NorthlandPositronics/Cogtrix/commit/23e593bd5d7a93e6968e02b500ad0568a3993b65))
- **wizard:** full-width YAML preview with syntax highlighting in Step 3 ([5245f12](https://github.com/NorthlandPositronics/Cogtrix/commit/5245f12abfdb971cb6ad48ef2582099b1ebe3ff4))
- **wizard:** full-width YAML preview with syntax highlighting in Step 3 ([aa88245](https://github.com/NorthlandPositronics/Cogtrix/commit/aa8824520f85f99ce1dd7b430accab1cf7f9e234))

### Bug Fixes

- **api/wizard:** resolve api_key from existing config when not submitted ([f781366](https://github.com/NorthlandPositronics/Cogtrix/commit/f7813663fed66d5268f73558ee8c512a1b6027e1))
- **api/wizard:** soft-fail first LLM invocation on provider errors ([ba1df0c](https://github.com/NorthlandPositronics/Cogtrix/commit/ba1df0cd40a663d9929a999bdd45fb61c5c4dd09))
- **api/wizard:** soft-fail probe errors so valid providers aren't blocked ([d2258b1](https://github.com/NorthlandPositronics/Cogtrix/commit/d2258b1a229832154462b758d3679078b06f4381))
- **api:** BUG-237/238/239/243/244/245/246/247 — provider CRUD, wizard, session safety ([7e0f0bb](https://github.com/NorthlandPositronics/Cogtrix/commit/7e0f0bbf88fc3647a957e1992b847784a651994c))
- **api:** replace deprecated HTTP_422_UNPROCESSABLE_ENTITY with HTTP_422_UNPROCESSABLE_CONTENT ([84317af](https://github.com/NorthlandPositronics/Cogtrix/commit/84317af88f53682f99831f811eb6cdf0c68005be))
- **api:** sentinel turn_task for sync=true 409 consistency; extract ToT final solution ([6ba904b](https://github.com/NorthlandPositronics/Cogtrix/commit/6ba904bcb885d35c39d7ff22f64f85f2c10fb245))
- **ci:** black format test_api_mcp_config_complete; remove unused variable in test_self_improving_loop_features ([a13e9f7](https://github.com/NorthlandPositronics/Cogtrix/commit/a13e9f7c063aaea54bde4175bdcec8d3d2fe6eb6))
- **cli:** shorten /help descriptions to fit 30-char column; soft-fail wizard probe ([c9ddc86](https://github.com/NorthlandPositronics/Cogtrix/commit/c9ddc861f9e1ffc3195b54460039669a710e6681))
- **cli:** start spinner before classify_think_task to eliminate silent pause ([5464266](https://github.com/NorthlandPositronics/Cogtrix/commit/5464266c759c81370c8fd6f0f70de3729811a83b))
- **docker:** restore --network host hint comment in entrypoint ([497f2c2](https://github.com/NorthlandPositronics/Cogtrix/commit/497f2c25a72fc676b750a339510e4242d8ddfe86))
- **docker:** skip wizard auto-start when CLI arguments are passed ([7c48b41](https://github.com/NorthlandPositronics/Cogtrix/commit/7c48b41f0197a6fe0e20855388abba75dbb2cbfb))
- **docker:** wizard auto-start must also check for .cogtrix.yml ([e234ba2](https://github.com/NorthlandPositronics/Cogtrix/commit/e234ba2be9679184c6c40159043a8abbebacee34))
- **orchestration:** atomic cache ops with single \_cache_lock ([158a33c](https://github.com/NorthlandPositronics/Cogtrix/commit/158a33c878b98f4196c03de2123d2557f559a5f1))
- **orchestration:** cancel timed-out futures to prevent zombie threads ([b143c62](https://github.com/NorthlandPositronics/Cogtrix/commit/b143c62e4cfa9f81a1884d76c9359f0a165dd6e1))
- **wizard:** clean up remaining user-facing log messages in setup_wizard ([0a65628](https://github.com/NorthlandPositronics/Cogtrix/commit/0a656282c3874bc9a7d677df9004f9c0c570c0b2))
- **wizard:** distinguish DNS failure from blocked address in \_list_ollama_models ([609251a](https://github.com/NorthlandPositronics/Cogtrix/commit/609251a8374e9091893f6017b2f15329b2d2b7c5))
- **wizard:** distinguish DNS failure from blocked address in \_list_ollama_models ([d83965c](https://github.com/NorthlandPositronics/Cogtrix/commit/d83965c4803bbc26b3ed4bd89580424932ea9391))
- **wizard:** remove internal bug IDs and jargon from Ollama error messages ([4abb907](https://github.com/NorthlandPositronics/Cogtrix/commit/4abb907cb87c8a9e143cb305df5854dfbeff9031))
- **wizard:** strip null/empty values from generated config before preview and write ([5ff6448](https://github.com/NorthlandPositronics/Cogtrix/commit/5ff64483c4969f865474c068850dc0c851a4fc74))

### Documentation

- **api:** update client contract and OpenAPI schema for [#94](https://github.com/NorthlandPositronics/Cogtrix/issues/94)/[#95](https://github.com/NorthlandPositronics/Cogtrix/issues/95) ([b13e7e7](https://github.com/NorthlandPositronics/Cogtrix/commit/b13e7e742dcb2e7f63b24fb13c6cd5cf00ea3819))
- **changelog:** add unreleased entries for [#94](https://github.com/NorthlandPositronics/Cogtrix/issues/94), [#95](https://github.com/NorthlandPositronics/Cogtrix/issues/95), BUG-215, BUG-002 ([48dc1d8](https://github.com/NorthlandPositronics/Cogtrix/commit/48dc1d8a43ad036bea7a806e95a849707755a21b))

## [0.1.35](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.34...v0.1.35) (2026-03-24)

### Features

- **wizard:** production model configuration after Step 1 bootstrap ([d6bad74](https://github.com/NorthlandPositronics/Cogtrix/commit/d6bad7404d39d1939be414987b1c87782497bb8d))
- **wizard:** production model configuration step after Step 1 bootstrap ([7a7b0e4](https://github.com/NorthlandPositronics/Cogtrix/commit/7a7b0e4750a479787587b40907e58272e1d8af9b))
- **wizard:** production model configuration, SSRF fixes, and end-to-end scenario tests ([d6bad74](https://github.com/NorthlandPositronics/Cogtrix/commit/d6bad7404d39d1939be414987b1c87782497bb8d))

### Bug Fixes

- **wizard:** allow RFC-1918 LAN addresses in user-typed Ollama URL ([bbcfce9](https://github.com/NorthlandPositronics/Cogtrix/commit/bbcfce95fe6d033b855956dcb07309b8265a68da))
- **wizard:** pass full bootstrap context to LLM — eliminate redundant questions and key leak ([aed59cd](https://github.com/NorthlandPositronics/Cogtrix/commit/aed59cd4b949007007525474fe18ad470d5d13c6))
- **wizard:** seed conversation with HumanMessage to fix vLLM/LiteLLM 400 error ([044071c](https://github.com/NorthlandPositronics/Cogtrix/commit/044071ceff057aab142a60acbfb2b0810d8312ad))
- **wizard:** stop \_extract_yaml at first closing fence to prevent Next-steps bleed ([fa54e55](https://github.com/NorthlandPositronics/Cogtrix/commit/fa54e55e838c2bc7d04bbfa4f5faf96403537cdb))

## [0.1.34](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.33...v0.1.34) (2026-03-24)

### Bug Fixes

- **api:** dead QueueFull catch and loop-local constant in confirmation/turn_runner ([57c2d95](https://github.com/NorthlandPositronics/Cogtrix/commit/57c2d952047e407565c69d8ed29b3b7536bdaca5))
- BUG-229/230/231/232/233/234/236 — Ollama SSRF guards, healthcheck sentinel, bound_cache lock, dedup key normalization ([5f7e2fe](https://github.com/NorthlandPositronics/Cogtrix/commit/5f7e2fe370391242f7c831ff3baf37230028a5fb))
- BUG-229/230/231/232/233/234/236 — Ollama SSRF guards, healthcheck sentinel, bound_cache lock, dedup key normalization ([e8fee43](https://github.com/NorthlandPositronics/Cogtrix/commit/e8fee4335dac4493e5466053aa1daf3fd58a9351))
- **wizard:** redact URL credentials in SSRF warning logs (CodeQL CWE-312) ([0afb74f](https://github.com/NorthlandPositronics/Cogtrix/commit/0afb74f8fc6d3b904a3e8b617d5630934de14abe))

### Documentation

- holistic accuracy audit — secondary guides and API spec verification ([89fcffa](https://github.com/NorthlandPositronics/Cogtrix/commit/89fcffaea228c9bb22b03fd817f3e948d56a53fb))
- holistic documentation audit — BUG-AUDIT-001/002 coverage + accuracy fixes ([ff7a6ec](https://github.com/NorthlandPositronics/Cogtrix/commit/ff7a6ec6b16fd26c83bd61debbd691007fc58539))

## [0.1.33](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.32...v0.1.33) (2026-03-24)

### Bug Fixes

- **wizard:** fail-fast connection test and clean error messages ([#82](https://github.com/NorthlandPositronics/Cogtrix/issues/82)) ([361c72c](https://github.com/NorthlandPositronics/Cogtrix/commit/361c72c807f10940c58413487f360f9c31d12b81))

### Documentation

- **api:** exhaustive accuracy audit of API documentation ([#81](https://github.com/NorthlandPositronics/Cogtrix/issues/81)) ([d5d6bfd](https://github.com/NorthlandPositronics/Cogtrix/commit/d5d6bfdbefca64e0e45562c132b3f305dfb64e87))

## [0.1.32](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.31...v0.1.32) (2026-03-23)

### Documentation

- DocsForge holistic documentation audit and update ([501b41a](https://github.com/NorthlandPositronics/Cogtrix/commit/501b41a3f51478bb7f585117d88083bd23d5042c))
- DocsForge holistic documentation audit and update ([f661b85](https://github.com/NorthlandPositronics/Cogtrix/commit/f661b853e40c991dba7b66bf9adad386bf72276c))
- DocsForge holistic documentation audit and update ([#78](https://github.com/NorthlandPositronics/Cogtrix/issues/78)) ([501b41a](https://github.com/NorthlandPositronics/Cogtrix/commit/501b41a3f51478bb7f585117d88083bd23d5042c))

## [0.1.31](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.30...v0.1.31) (2026-03-23)

### Features

- **api:** route --debug logs to stdout/stderr instead of file ([d8778ba](https://github.com/NorthlandPositronics/Cogtrix/commit/d8778ba41792abd428273784bcb1e5a27675e604))
- **api:** route --debug logs to stdout/stderr instead of file ([cbb12cc](https://github.com/NorthlandPositronics/Cogtrix/commit/cbb12cc1741026cfe139c8984399386fcd206293))
- **api:** route --debug logs to stdout/stderr instead of file ([#75](https://github.com/NorthlandPositronics/Cogtrix/issues/75)) ([d8778ba](https://github.com/NorthlandPositronics/Cogtrix/commit/d8778ba41792abd428273784bcb1e5a27675e604))

## [0.1.30](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.29...v0.1.30) (2026-03-23)

### Bug Fixes

- **docker:** fix alembic invocation and Python version alignment ([25950cd](https://github.com/NorthlandPositronics/Cogtrix/commit/25950cd055226038c42fa0aabdc36e1edeb0c526))
- **docker:** fix alembic invocation and Python version alignment ([78e94f9](https://github.com/NorthlandPositronics/Cogtrix/commit/78e94f9b286cb2de6b6d6d4c62f3b7e91a14087c))
- **docker:** fix alembic invocation and Python version alignment ([#70](https://github.com/NorthlandPositronics/Cogtrix/issues/70)) ([25950cd](https://github.com/NorthlandPositronics/Cogtrix/commit/25950cd055226038c42fa0aabdc36e1edeb0c526))
- replace f-string logging with lazy %-formatting (11 sites) ([e61c853](https://github.com/NorthlandPositronics/Cogtrix/commit/e61c8530a4dad4764b1555a11e3e931d54f5e8dd))
- replace f-string logging with lazy %-formatting (11 sites) ([#72](https://github.com/NorthlandPositronics/Cogtrix/issues/72)) ([e61c853](https://github.com/NorthlandPositronics/Cogtrix/commit/e61c8530a4dad4764b1555a11e3e931d54f5e8dd))
- replace f-string logging with lazy %-formatting across 11 call sites ([9e95182](https://github.com/NorthlandPositronics/Cogtrix/commit/9e95182b7e9d6403051f7b0dfd250bb0e1eb7ca5))

## [0.1.29](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.28...v0.1.29) (2026-03-23)

### Bug Fixes

- **api:** correct misleading errors across sessions, assistant, and knowledge routes ([f85f1d6](https://github.com/NorthlandPositronics/Cogtrix/commit/f85f1d678140c0a2e79f9a8d1edac5a3b321ede5))
- **api:** correct misleading errors across sessions, assistant, and knowledge routes ([32fd02c](https://github.com/NorthlandPositronics/Cogtrix/commit/32fd02c2a9bd8b03311a7303447cade3f91c35fc))
- **api:** propagate real provider error in wizard step 0/1 (BUG-001) ([affd3ec](https://github.com/NorthlandPositronics/Cogtrix/commit/affd3ec0cbe8507a9d80f8a63e8898182dce9d45))
- **api:** propagate real provider error in wizard step 0/1 (BUG-001) ([7bfaf3c](https://github.com/NorthlandPositronics/Cogtrix/commit/7bfaf3c4ef3453d68e6fcaa80fc06119e9b8a423))
- **api:** propagate real provider error in wizard step 0/1 (BUG-001) ([#51](https://github.com/NorthlandPositronics/Cogtrix/issues/51)) ([affd3ec](https://github.com/NorthlandPositronics/Cogtrix/commit/affd3ec0cbe8507a9d80f8a63e8898182dce9d45))
- **docker:** holistic Dockerfile, compose, and entrypoint audit ([#48](https://github.com/NorthlandPositronics/Cogtrix/issues/48)) ([6ccad8c](https://github.com/NorthlandPositronics/Cogtrix/commit/6ccad8c709d795ac606361a1d234d0f92dfab14d))
- **graph:** replace mid-conversation SystemMessage with HumanMessage ([6e109ce](https://github.com/NorthlandPositronics/Cogtrix/commit/6e109ce85ec57006d4f8a1c96474bf538448a5dd))
- **graph:** replace mid-conversation SystemMessage with HumanMessage ([#50](https://github.com/NorthlandPositronics/Cogtrix/issues/50)) ([13f58d3](https://github.com/NorthlandPositronics/Cogtrix/commit/13f58d37f37de4de1faf32f7b224976c74837fdf))
- reset agent_state on pipeline-phase cancel; gate debug-log computation ([#40](https://github.com/NorthlandPositronics/Cogtrix/issues/40)) ([32b7285](https://github.com/NorthlandPositronics/Cogtrix/commit/32b72857d599bf52565b415a272fa41e2a3fe7a6))
- resolve all forge audit findings — lazy logging, monotonic clock, and async I/O ([6d095ed](https://github.com/NorthlandPositronics/Cogtrix/commit/6d095ed85c66ad861cf6ec00f53c7b817d863bdb))

### Documentation

- **api:** exhaustive accuracy audit of client-contract.md ([a2d56f1](https://github.com/NorthlandPositronics/Cogtrix/commit/a2d56f1bd93816786a605719a4f70796652be1c1))
- **api:** exhaustive accuracy audit of client-contract.md ([3a5fb6b](https://github.com/NorthlandPositronics/Cogtrix/commit/3a5fb6b3ac4806b3d3c8c7e488a8f9bcc405c890))
- **api:** exhaustive accuracy audit of client-contract.md ([#46](https://github.com/NorthlandPositronics/Cogtrix/issues/46)) ([602e76d](https://github.com/NorthlandPositronics/Cogtrix/commit/602e76d22bc8c87e3c5c609dcab855359032ad22))
- **api:** fix WorkflowDocumentOut TypeScript type and delete path ([#36](https://github.com/NorthlandPositronics/Cogtrix/issues/36)) ([d2fae0b](https://github.com/NorthlandPositronics/Cogtrix/commit/d2fae0be46fd64458637534717a944466b32f976))
- **api:** regenerate openapi.yaml and openapi.json from live routes ([a9ee98b](https://github.com/NorthlandPositronics/Cogtrix/commit/a9ee98b774c775dfcff0378127d2cfc337ea6ed1))
- **api:** regenerate openapi.yaml and openapi.json from live routes ([6476a55](https://github.com/NorthlandPositronics/Cogtrix/commit/6476a55a7b39cd33600f7b09192f78198b1bb082))
- **architecture:** update turn_runner and callbacks descriptions for cancel handling and debug guard ([#44](https://github.com/NorthlandPositronics/Cogtrix/issues/44)) ([5c1f56b](https://github.com/NorthlandPositronics/Cogtrix/commit/5c1f56bced153ec21d3ae9c5649a1a992f3e8cf1))
- holistic documentation audit — sync with recent fixes ([0a45796](https://github.com/NorthlandPositronics/Cogtrix/commit/0a45796c2f5e30dd4a37b7128c0242b12cc3a640))
- holistic documentation audit — sync with recent fixes ([8da7dcb](https://github.com/NorthlandPositronics/Cogtrix/commit/8da7dcbf1359b40f75aeb3e82d043176e348b341))

## [0.1.28](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.27...v0.1.28) (2026-03-22)

### Bug Fixes

- xfail ws reconnect test teardown hang ([53edca6](https://github.com/NorthlandPositronics/Cogtrix/commit/53edca6651d27f3ef6d0ff9b02b5c0e3c7b18c2b))

## [0.1.27](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.26...v0.1.27) (2026-03-22)

### Bug Fixes

- add WorkflowDocumentOut schema to workflow document endpoints ([#30](https://github.com/NorthlandPositronics/Cogtrix/issues/30)) ([ce1f958](https://github.com/NorthlandPositronics/Cogtrix/commit/ce1f958ae3bcbda7db45c408e49e91d65960391a))
- forge audit findings — test timeout + type annotation ([5945af8](https://github.com/NorthlandPositronics/Cogtrix/commit/5945af868d7e493dcb1dac303b528e0201e234e2))
- **wizard:** structured error on connection failure + remove protocol leakage ([ab67912](https://github.com/NorthlandPositronics/Cogtrix/commit/ab67912d17653dd85fc749714d92d8207b0b5721))

### Documentation

- **api:** holistic API documentation audit — 9 issues fixed ([#29](https://github.com/NorthlandPositronics/Cogtrix/issues/29)) ([6e4dc1c](https://github.com/NorthlandPositronics/Cogtrix/commit/6e4dc1c80f270b0e095a06b0e42d7c73d7fc85c1))
- holistic general documentation audit — 6 corrections ([#31](https://github.com/NorthlandPositronics/Cogtrix/issues/31)) ([6891847](https://github.com/NorthlandPositronics/Cogtrix/commit/6891847f5936429bb4a3ed7c2172f4bf1387ffcf))

## [0.1.26](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.25...v0.1.26) (2026-03-22)

### Features

- **api:** add DEBUG-level logging to core chat pipeline ([5f266f2](https://github.com/NorthlandPositronics/Cogtrix/commit/5f266f2e259b3b2533ecaf1a49d14c0c5c6540ef))
- **api:** add DEBUG-level logging to core chat pipeline ([05b8c5d](https://github.com/NorthlandPositronics/Cogtrix/commit/05b8c5d28690228bbe3c160d38bd839723a737ba))

### Documentation

- update documentation for v0.1.24–v0.1.25 changes ([49750c1](https://github.com/NorthlandPositronics/Cogtrix/commit/49750c1110aa469cd2a456f4a547cfffeb4c616f))
- update documentation for v0.1.24–v0.1.25 changes ([7a71bc3](https://github.com/NorthlandPositronics/Cogtrix/commit/7a71bc309a05b3ef4371f635708a01ae6c4b8de8))

## [0.1.25](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.24...v0.1.25) (2026-03-22)

### Bug Fixes

- **docker:** add socket timeout and explicit status check to healthcheck ([#17](https://github.com/NorthlandPositronics/Cogtrix/issues/17)) ([2baea70](https://github.com/NorthlandPositronics/Cogtrix/commit/2baea7075d019f171a0aaa50f2d0d082f28b4e7e))
- **docker:** healthcheck with 4s socket timeout and explicit status check ([5204cab](https://github.com/NorthlandPositronics/Cogtrix/commit/5204cab6bb5181a786dc9bb230e011d736b213eb))
- **wizard:** correctly drain escape sequences and read multi-byte UTF-8 in masked input ([4393c54](https://github.com/NorthlandPositronics/Cogtrix/commit/4393c54795538ac03421db8be221bb9ee192f167))
- **wizard:** correctly drain escape sequences and read multi-byte UTF-8 in masked input ([e104117](https://github.com/NorthlandPositronics/Cogtrix/commit/e1041171d5781751cd08552b1ded887027ee5adb))

## [0.1.24](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.23...v0.1.24) (2026-03-22)

### Features

- wizard UX improvements, vLLM no-auth fix, entrypoint fix ([758d703](https://github.com/NorthlandPositronics/Cogtrix/commit/758d70319d2a536c75b30e9e77a46ba7073e2191))
- **wizard:** masked API key input, retry defaults, empty key support ([#14](https://github.com/NorthlandPositronics/Cogtrix/issues/14)) ([2a229de](https://github.com/NorthlandPositronics/Cogtrix/commit/2a229de272d55196d8628396d505f464b4fa9b95))

### Bug Fixes

- **docker:** always start wizard when no config/key regardless of args ([2a229de](https://github.com/NorthlandPositronics/Cogtrix/commit/2a229de272d55196d8628396d505f464b4fa9b95))
- **providers/openai:** pass 'no-key' placeholder for unauthenticated endpoints ([2a229de](https://github.com/NorthlandPositronics/Cogtrix/commit/2a229de272d55196d8628396d505f464b4fa9b95))

### Documentation

- **index:** add Repositories & Packages section with GHCR links ([2a229de](https://github.com/NorthlandPositronics/Cogtrix/commit/2a229de272d55196d8628396d505f464b4fa9b95))

## [0.1.23](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.22...v0.1.23) (2026-03-20)

### Bug Fixes

- **ci:** post status checks to correct SHA after Contents API commit ([d18249c](https://github.com/NorthlandPositronics/Cogtrix/commit/d18249c3f79ae0606daea1c3aefbda2fb604d7ed))
- **ci:** post status checks to the new commit SHA after Contents API push ([a63e146](https://github.com/NorthlandPositronics/Cogtrix/commit/a63e14675f95b8cede988917a6bd53ca813309db))

## [0.1.22](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.21...v0.1.22) (2026-03-20)

### Bug Fixes

- **ci:** sign uv.lock update commit via GitHub Contents API ([#27](https://github.com/NorthlandPositronics/Cogtrix/issues/27)) ([ba4ccc1](https://github.com/NorthlandPositronics/Cogtrix/commit/ba4ccc1ee1ae7e2a764aa16a21d57a2fec4d8a52))
- **ci:** use jq --rawfile to avoid ARG_MAX on base64-encoded uv.lock ([#34](https://github.com/NorthlandPositronics/Cogtrix/issues/34)) ([61b68c3](https://github.com/NorthlandPositronics/Cogtrix/commit/61b68c350db494d69076ec178971da55c99a6dc1))
- **ci:** use jq temp file to avoid arg-too-long on large uv.lock ([#30](https://github.com/NorthlandPositronics/Cogtrix/issues/30)) ([a7a6733](https://github.com/NorthlandPositronics/Cogtrix/commit/a7a67334aecb1a7eee367a78a88e2b8a8cb0d9e5))
- **deps:** resolve 7 security vulnerabilities in dependencies ([#32](https://github.com/NorthlandPositronics/Cogtrix/issues/32)) ([94cf6d5](https://github.com/NorthlandPositronics/Cogtrix/commit/94cf6d570403c707a0f61f12fb5d79aea80383c4))
- use jq --rawfile to bypass ARG_MAX on large uv.lock ([b05d0a1](https://github.com/NorthlandPositronics/Cogtrix/commit/b05d0a10fe69f44fd114b46eda321131917425c9))

## [0.1.21](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.20...v0.1.21) (2026-03-20)

### Bug Fixes

- **ci:** add --extra api and timeout to release workflow ([34087f6](https://github.com/NorthlandPositronics/Cogtrix/commit/34087f681b7189236bea7834284e26ed516e459c))
- **ci:** add --extra api and timeout to release workflow; make Docker publish advisory ([d8c76f7](https://github.com/NorthlandPositronics/Cogtrix/commit/d8c76f7bce971308d05568000bbda08e4cecbf8f))
- **ci:** add statuses: write permission to release-please workflow ([4115270](https://github.com/NorthlandPositronics/Cogtrix/commit/41152706924387bcc6a1e604f7f75e4610bb0c6a))
- **ci:** add statuses: write permission to release-please workflow ([ef30656](https://github.com/NorthlandPositronics/Cogtrix/commit/ef3065623cbd44f8c8f9ad11a9cd69503c98c86a))
- **ci:** add statuses: write permission to release-please workflow ([024f167](https://github.com/NorthlandPositronics/Cogtrix/commit/024f167c1607b33ee750c1575447b99a7396d5ba))

## [0.1.20](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.19...v0.1.20) (2026-03-20)

### Features

- **ci:** guard main source branch, fix CI dependencies and hanging tests ([2a96cb7](https://github.com/NorthlandPositronics/Cogtrix/commit/2a96cb7d8efa96cbcda15a755585914c25a1b68d))
- **ci:** guard main source branch, fix CI dependencies and hanging tests ([06dba98](https://github.com/NorthlandPositronics/Cogtrix/commit/06dba98bfc5a91393f8a88fd4cfd33286ad18634))
- **ci:** guard main source branch, fix CI dependencies and hanging tests ([#17](https://github.com/NorthlandPositronics/Cogtrix/issues/17)) ([2a96cb7](https://github.com/NorthlandPositronics/Cogtrix/commit/2a96cb7d8efa96cbcda15a755585914c25a1b68d))

## [Unreleased]

### Breaking Changes

- **config:** Provider/model separation refactor — `ProviderConfig` now holds connection info only (`type`, `base_url`, `api_key`, `tool_instructions`); all inference parameters (`model`, `temperature`, `num_ctx`, `max_tokens`) belong exclusively in `ModelConfig`. `models.default` selects the active model alias. Legacy top-level `provider`/`model` keys and model fields inside `providers:` entries
  are auto-migrated but deprecated.
- **cli:** `--provider` / `-p` CLI flag removed; use `--model` / `-m` with a model alias instead
- **cli:** `/provider` command is now read-only (lists providers); use `/model` to switch models
- **config:** `COGTRIX_PROVIDER` environment variable removed; use `COGTRIX_MODEL` instead
- **api:** `POST /config/provider` endpoint removed (returns 410 Gone)

### Features

- **assistant:** Level 1 outbound messaging — `POST /api/v1/assistant/outbound` admin endpoint sends operator-initiated messages to phonebook contacts via the agent pipeline (bypasses input guardrails, applies output guardrails, updates memory)
- **assistant:** Level 2 campaign system — multi-contact outbound campaigns with per-target progress tracking, automatic follow-ups when contacts don't reply, escalation after max attempts, and agent-classified goal completion via `report_campaign_outcome` tool; 6 API endpoints for CRUD + launch; persistence to `data/assistant/campaigns.json`; background follow-up thread with configurable check
  interval
- **tools:** two-tier tool loading — agent-loaded tools auto-unload after each prompt cycle; manually loaded tools (via `/tools load`, `--activate-tools`, or API `PATCH`) are pinned and persist until explicitly unloaded
- **cli:** `--activate-tools LIST` flag pins comma-separated tools as active at startup
- **cli:** `/tools unload <name>` command to unpin and return a tool to the on-demand pool
- **rag:** `query_knowledge_base` auto-activates (pinned) when a FAISS knowledge base exists; dynamic description shows index count and size
- **rag:** multi-index search — queries both global CLI index and per-document API indexes, merges and deduplicates results
- **api:** tool status now includes `"pinned"` for manually loaded tools in `ToolStatus` enum; API version bumped to 1.1.0
- **api:** REST + WebSocket API layer with JWT authentication, session management, streaming agent turns, tool management, memory control, RAG document endpoints, config management, MCP server management, and assistant mode control (65 REST endpoints + 2 WebSocket streams)
- **api:** API key authentication (`cgx_live_` prefix) for programmatic and CI access
- **api:** setup wizard API for interactive configuration via HTTP
- **api:** live log streaming via WebSocket at `ws://host/ws/v1/logs` (admin only)
- **config:** `Config.resolve_llm_config()` and `resolve_llm_config_for(alias)` — new primary LLM resolution methods returning `(ProviderConfig, ModelConfig)` tuples
- **providers:** `create_chat_model_from_configs(provider_config, model_config)` — new dual-config LLM factory replacing the old single-config path
- **config:** `_parse_providers_section()` auto-migrates model fields from provider entries to the models registry for backward compatibility
- **whatsapp:** track locally-archived chats in `_locally_archived` set to prevent re-processing after WhatsApp auto-unarchives (BUG-113)

### Bug Fixes

- **campaign:** `Campaign.from_dict` no longer mutates the caller's dict via `pop()` — uses `get()` instead (BUG-221)
- **campaign:** `on_reply` releases the lock before calling `save()` to avoid blocking other threads during disk I/O (BUG-222)
- **campaign:** `_process_follow_ups` re-checks `target.status` under lock at the escalation branch to prevent racing with concurrent `mark_target_outcome` (BUG-223)
- **campaign:** `launch()` sets target to `"active"` before `handle_outbound` call so `on_reply()` can match replies arriving during the send window (BUG-224)
- **campaign:** `start()` validates handler is wired via `set_handler()` and guards thread creation under the lock to prevent duplicate follow-up threads (BUG-225)
- **api:** campaign CRUD routes (`create`, `update`, `delete`) now wrapped with `asyncio.to_thread` to prevent blocking the event loop during file I/O (BUG-226)
- **api:** `_validate_campaign_id` enforces UUID regex on all campaign path parameters to prevent injection (BUG-227)
- **api:** `status_filter` query param on `GET /campaigns` typed as `CampaignStatus` for Pydantic validation (rejects invalid values with 422)
- **api:** `_resolve_contact` extracted as unified phonebook lookup — `send_outbound` now prefers active channels (matching campaign target resolution behavior)
- **api:** `stop_assistant` route uses `executor.shutdown(wait=True)` to drain in-flight agent turns before `session_mgr.save_all()`, eliminating a race between executor threads and memory persistence (data-loss fix; mirrors `service.py` `_handle_shutdown` behaviour)
- **api:** `WebSocketCallbackHandler` now tracks `tool_call_count` and `_extract_token_counts` in `turn_runner.py` returns it, so the `done` WebSocket message reports actual tool invocations instead of a hardcoded `0`
- **api:** atomic `INSERT…SELECT` for admin role election in `create_with_role_election` eliminates registration race condition
- **api:** per-session `asyncio.Event` in `ApiSessionRegistry._pending` prevents duplicate `warm_session` calls for concurrent requests targeting the same session (TOCTOU fix)
- **api:** DB session threaded through route `Depends(get_db)` into auth helpers, eliminating redundant database connections
- **api:** fix silent token degradation in `get_current_user_optional` — supplied tokens that are expired or invalid now re-raise instead of falling through to anonymous access (P0 security fix)
- **api:** catch `IntegrityError` in registration endpoint for concurrent duplicate username/email submissions
- **api:** bulk `DELETE` for `clear_history` with `keep_last` parameter (performance)
- **api:** protect `_cancel_requested` flag with lock in `ApiConfirmationUI.cancel()` (thread safety)
- **api:** fix `WSLogHandler` crash, `turn_runner` blocking save, and RAG flush/delete without commit
- **api:** path traversal guard in RAG upload, WebSocket close ordering, event loop blocking in turn runner
- **assistant:** resolve BUG-091 through BUG-112 across deferral system and assistant subsystem
- **api:** reset `_cancel_requested = False` at the start of `render_prompt()` so cancellation from a previous turn does not silently deny all future tool confirmations (P0)
- **api:** `_validate_doc_id()` UUID regex guard in RAG endpoints prevents path traversal via document ID parameters
- **api:** `_snapshot_sessions()`, `_snapshot_scheduler_queue()`, `_snapshot_deferral_records()` helpers copy dicts under lock before iteration, eliminating five race conditions in assistant route handlers
- **api:** `warm_session()` in `session_bridge.py` wraps `_build_memory_manager` and `_build_llm` with `asyncio.to_thread` to prevent blocking the event loop during session warm-up
- **api:** `clear_memory` and `switch_memory_mode` in `memory.py` route blocking `mm.clear()`, `old_mm.save()`, and `new_mm.load()` calls through `asyncio.to_thread`
- **api:** `ConnectionManager.connect()` in `ws.py` releases `_lock` before closing the displaced WebSocket connection to avoid holding the lock across I/O
- **api:** `stop_assistant()` wraps blocking service shutdown calls with `asyncio.to_thread`
- **api:** RAG document list endpoint uses compound `(created_at, id)` keyset cursor for stable pagination ordering; `_doc_to_out` disk I/O (file stat) runs via `asyncio.to_thread` when paginating
- **api:** deleting a session now calls `manager.disconnect(session_id)` to close any orphaned WebSocket connection before archiving the record
- **api:** `get_chat_messages` correctly derives message count by calling `get_messages()` when available (was always returning 0)
- **api:** fix agent amnesia in `turn_runner._build_history()` — `prepare_context()` was called with no arguments (silently raising `TypeError`) and its return value was accessed via `.get()` on a dataclass (silently raising `AttributeError`), causing every turn to start with empty history; fixed by forwarding `user_input` and accessing `.messages` attribute directly (P0)
- **api:** `asyncio.CancelledError` in `run_message_turn()` was swallowed without re-raising, breaking `asyncio.Task.cancel()` semantics; fixed by adding `raise` after cleanup (P0)
- **api:** `get_or_warm()` in `session_bridge.py` now saves the discarded `ApiSession` when a concurrent warmer wins the race, preventing a memory manager resource leak (P1)
- **api:** `memory_manager.update()` and `.save()` in `turn_runner` now run via `asyncio.to_thread` to prevent blocking the event loop on the threading lock and file I/O (P1)
- **api:** `_http_exception_handler` now maps non-dict exception detail to a status-appropriate error code via `_STATUS_CODE_MAP` instead of always returning `code="INTERNAL_ERROR"` for 4xx responses (P1)
- **api:** `check_provider_health` in `routes/config.py` now runs `create_chat_model_from_configs()` via `asyncio.to_thread` to prevent network I/O stalling the event loop (P1)
- **api:** `reload_config` in `routes/config.py` now runs `Config()` file I/O via `asyncio.to_thread` (P1)
- **api:** fix `MemoryUpdatePayload.tokens_used` schema example — was `"1200"` (string) for an `int` field, producing a malformed OpenAPI schema (P2)
- **api:** fix fake-lock data races in `assistant.py` routes — `remove_from_blacklist`, `list_knowledge`, `search_knowledge`, and `delete_fact` created anonymous `threading.Lock()` as fallback instead of using the actual object lock, providing zero mutual exclusion against concurrent readers or writers; all four routes now acquire the real `violation_tracker._lock` or `knowledge_store._lock` (P0)
- **api:** `delete_fact` in `assistant.py` contained dead unreachable code after `raise HTTPException` refactor — removed (P1)
- **api:** `start_assistant` in `assistant.py` now creates the LLM via `asyncio.to_thread(create_chat_model_from_configs, ...)` instead of calling it synchronously in the async handler, preventing event loop blocking on provider initialization (P1)
- **api:** `patch_session` in `sessions.py` now calls `_build_llm` via `asyncio.to_thread` when provider or model changes, preventing event loop blocking on LLM initialization (P1)
- **api:** `_config_to_out` in `config.py` refactored — new `_read_raw_yaml()` async helper offloads config file I/O to a thread pool via `asyncio.to_thread`; `_config_to_out` signature changed from `is_admin: bool` to `raw_yaml: str | None` so the async I/O happens in the caller (P1)
- **api:** `run_message_turn()` now updates `session.last_activity` at the end of each successful turn so the 30-minute idle eviction TTL resets correctly — previously a long-running agent turn would age out the session mid-execution, causing the next request to re-warm from DB (BUG-120)
- **api:** `run_message_turn()` now fully implements `mode='think'` and `mode='delegate'` — think mode wires `classify_think_task` → optional research delegate → `force_deep_think`; delegate mode wires `force_delegation` with parallel sub-agent execution; all blocking LLM calls run via `asyncio.to_thread`; new `agent_state` values (`analyzing`, `deep_thinking`, `researching`, `delegating`) stream
  progress to the frontend (BUG-122)
- **assistant:** removed dead conditional guard in `MessageHandler._run_agent()` — the `if defer_state/suppress_state` branch and its fallthrough were identical; the guard was a no-op that added confusion without preventing any unintended behaviour (BUG-121)
- **security:** remove `copy` from `SAFE_MODULES` in `python_exec.py` — `copy.deepcopy` invokes `__reduce_ex__` via C code, bypassing sandbox attribute guards (SEC-01)
- **security:** cap unbounded `.*` in 2 guardrails injection patterns to `.{0,200}` to prevent ReDoS on attacker-controlled assistant input (SEC-02)
- **security:** cap unbounded `.*?` in 3 `DEEP_THINK_TRIGGERS` patterns and 2 `DELEGATION_TRIGGERS` patterns to bounded `.{0,80}?` / `.{3,80}?` to prevent ReDoS (SEC-03/04)
- **security:** `resolve_data_path()` now returns the resolved absolute path, closing a TOCTOU window for symlink attacks between validation and file open (SEC-05)
- **shell:** switch from `subprocess.run` to `Popen` with `start_new_session=True` + `os.killpg()` on timeout — kills the entire process group instead of just the direct `/bin/sh` child, preventing orphaned grandchild processes
- **delegate:** fix `_validate_json_response` fence stripping — now finds the matching closing fence after the opening ` ```json ` instead of unconditionally stripping the last triple-backtick, which corrupted responses containing additional code blocks
- **delegate:** change `circuit_breaker.check_availability()` to `circuit_breaker._check_availability_locked()` inside `with _circuit_breaker_lock:` block to fix redundant reentrant lock acquisition
- **calculator:** tighten `_safe_pow` guard from `exp >= 10_000` to `abs(exp) >= 1_000` — prevents `9999**9999` (39K-digit computation) and catches negative exponents
- **graph:** parallel tool `future.result()` now has a 10-minute timeout — on timeout an error `ToolMessage` is produced instead of hanging indefinitely (BUG-202)
- **graph:** `_detect_tool_request` normalizes bare string args to single-element lists so `{"add": "web_search"}` works the same as `{"add": ["web_search"]}` (BUG-204)
- **graph:** auto-expansion dedup cache key now uses the resolved (post-fuzzy-match) tool name for correct cross-name deduplication
- **api:** `run_message_turn()` calls `reset_for_new_prompt()` at turn start to clear ephemeral tools and `deny_all`, matching CLI prompt boundaries (BUG-198)
- **api:** `warm_session()` populates `session_state.all_tool_originals` from tool registry so unload/disable can restore canonical tool objects (BUG-199)
- **api:** `patch_session_tools` acquires `turn_lock` around all `run_config` mutations to prevent races with in-flight agent turns (BUG-196)
- **api:** `_classify_tool_status` only reports `"auto_approved"` when the tool is also in `loaded_tools` — an approval on an on-demand tool does not imply it is active
- **cli:** `/tools disable` now removes from both `pinned_tools` and `loaded_tools` in addition to adding to `denials` (BUG-197)
- **rag:** `_has_faiss_index()` validates actual FAISS index files exist before adding directories to the search list (BUG-200)
- **rag:** `_collect_faiss_dirs()` applies containment check on resolved `idx` path to prevent symlink traversal via intermediate components (BUG-191)
- **rag:** multi-index search uses `similarity_search_with_score` with score-based sorting for cross-index relevance ranking (BUG-193)
- **delegate:** `future.cancel()` now called in the `remaining <= 0` timeout branch to prevent leaked LLM threads (BUG-195)
- **compression:** warning log emitted on compression LLM timeout before truncation fallback
- **compression:** `as_completed()` now receives a pool-level timeout so hung LLM calls trigger truncation fallback instead of blocking the agent turn indefinitely (BUG-207)
- **api:** error messages in `run_message_turn()` use `put_nowait` (not blocking `await put()`) and the `done` message uses `asyncio.wait_for(put(), timeout=5.0)` to prevent deadlock on bounded queue in REST-only sessions (BUG-209)
- **rag:** `knowledge_base_stats()` and `_build_description()` wrap `iterdir()`/`stat()` calls with `OSError` handling to survive permission errors and TOCTOU races on FAISS directories (BUG-211)
- **tools:** `configure_rag_tool()` catches `(ImportError, OSError)` instead of just `ImportError` so a broken FAISS directory does not crash startup (BUG-211)
- **cleanup:** remove dead `_has_phantom_tool_call` alias and unused import from `cogtrix.py` (BUG-206)
- **api:** all workflow API `{workflow_id}` path parameters validated against `^[a-zA-Z0-9][a-zA-Z0-9_-]*$` regex at the route boundary before reaching filesystem operations (BUG-212)
- **api:** `update_workflow` route uses `dataclasses.replace()` to build a copy instead of mutating the live registry object, preventing concurrent readers from seeing partially-updated state (BUG-213)
- **assistant:** `_load_prompt_from_value` now resolves relative paths (e.g. `./`, `../`) against `data_dir` with `is_relative_to` containment, closing a path traversal bypass (BUG-214)
- **api:** `upload_workflow_document` validates resolved file path stays inside `data_dir` via `is_relative_to` before writing (BUG-216)
- **api:** `on_llm_new_token` sets `final=True` only when `tool_call_count > 0 AND len(_tool_starts) == 0`, preventing premature final-response marking during intermediate tool reasoning (BUG-218)
- **compression:** inner `future.result()` now has `timeout=120` so the per-future `TimeoutError` handler is reachable (was dead code without a timeout) (BUG-219)
- **assistant:** `_auto_detect` returns the highest-scoring workflow that meets `min_confidence`, not the first alphabetical match (BUG-220)
- **api:** `update_workflow` route returns `_wf_to_out(updated)` instead of the stale pre-update object
- **graph:** `handle_phantom` and `process_tools` nodes now inject mid-conversation guidance as `HumanMessage` instead of `SystemMessage`, fixing compatibility with providers that reject `SystemMessage` outside position 0 (Qwen3, strict vLLM deployments)
- **api:** `patch_session` in `sessions.py` validates the requested model alias via `config.resolve_llm_config_for()` before touching the DB — invalid alias now returns 422 `MODEL_NOT_FOUND` instead of silently committing an unusable config
- **api:** `start_assistant` in `assistant.py` returns 503 `SERVICE_UNAVAILABLE` (not 409 `CONFLICT`) when server configuration or tool registry is absent at startup, or when service initialization fails; 409 `ASSISTANT_ALREADY_RUNNING` is now reserved exclusively for the case where the service is already running
- **api:** `violation_tracker.save()`, `knowledge_store.save()`, `scheduler.edit_message()`, and `scheduler.cancel_message()` in `assistant.py` routes are now wrapped with `asyncio.to_thread`, preventing event loop blocking during JSON persistence (P2)
- **callbacks:** `WebSocketCallbackHandler` now uses `time.monotonic()` for tool duration measurement, consistent with `ToolCallLogger` in `runner.py` and immune to NTP clock-adjustment drift
- **wizard:** `_test_connection()` propagates real provider errors to the API caller (BUG-001) — step 1 LLM initialization failure now returns 422 instead of 500
- **logging:** 46 f-string log calls converted to lazy `%`-formatting across `src/` — eliminates string evaluation cost when the log level is filtered out

### Features

- **assistant:** workflow system — `WorkflowRegistry` loads YAML workflow definitions from `data/workflows/<id>/workflow.yaml`; each workflow bundles a system prompt, per-workflow FAISS knowledge base, and tool policy; chat-to-workflow bindings persisted in `data/workflows/bindings.json`; resolution order: explicit binding → contact_prompts fallback → auto-detect (keyword/regex scoring) → global
  default; API CRUD at `/api/v1/assistant/workflows/` (11 endpoints)
- **api:** user management — 4 admin-only endpoints: list all users, create user, update role, delete user; `UserRepository` extended with `list_all()`, `update_role()`, `delete()` methods
- **api:** `TokenPayload.final` boolean field distinguishes preamble tokens (before tool calls) from the final response after all tools complete
- **api:** `SessionCreateRequest.name` auto-generates `"Session YYYY-MM-DD HH:MM"` via `default_factory`
- **api:** `ConfigOut` now includes `system_prompt` and `guardrails` fields for WebUI consumption
- **api:** `_run_think_pipeline` checks `session.cancel_event.is_set()` between pipeline phases (classify → research → deep_think) to avoid proceeding to expensive phases after cancel
- **config:** auto-migrated model aliases use `"{provider}/{model}"` format (e.g. `"openai/gpt-4.1-mini"`) instead of bare provider name

### Performance

- **compression:** convert eager `_COMPRESSION_POOL` (4 threads spawned at import) to lazy `_get_compression_pool()` with double-checked locking — threads only created when compression actually runs (PERF-01)
- **runner:** fix `ToolCallLogger._evict_stale` calling `time.monotonic()` twice — reuses `now` parameter for cutoff calculation, eliminating a redundant syscall (PERF-02)
- **intent:** hoist `_cat_by_name` dict comprehension to module-level `_THINK_CAT_BY_NAME` — avoids rebuilding a 23-entry dict on every `classify_think_task()` call (PERF-03)

### Build

- add `api` optional dependency group to `pyproject.toml` (FastAPI, uvicorn, SQLAlchemy async, aiosqlite, alembic, python-jose, passlib)
- update `Dockerfile` for API mode support (uvicorn, alembic migrations at startup)
- update `docker-entrypoint.sh` with `api` / `--api` mode that runs `alembic upgrade head` then starts uvicorn

## [0.1.19](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.18...v0.1.19) (2026-03-05)

### Features

- **api:** assistant auto-start on API server boot ([838dfd6](https://github.com/NorthlandPositronics/Cogtrix/commit/838dfd6b11ac4f3e523cb0f4e4cd9764c18f0671))
- **api:** assistant auto-start on API server boot ([5c361da](https://github.com/NorthlandPositronics/Cogtrix/commit/5c361dab74d297b8d029c55fed6443c5c8aaa73c))

### Bug Fixes

- BUG-118 protect compress_tool_message against prompt injection ([93a7c9c](https://github.com/NorthlandPositronics/Cogtrix/commit/93a7c9c4159f61eb2abbfdfbcd27d624ee3cf11a))
- move Pydantic serialisation outside asyncio.Lock in ConnectionManager.send ([df78f2e](https://github.com/NorthlandPositronics/Cogtrix/commit/df78f2eb04356fa34c3571053a2825e0fc0fe59a))
- ProjectForge audit sprint 1 — 9 API bugs + 1 security fix ([5f6c5f6](https://github.com/NorthlandPositronics/Cogtrix/commit/5f6c5f6e26ef5691a3c28303bc90ea7e04d56282))
- resolve BUG-115 and BUG-117 in API turn runner concurrency ([36f1d35](https://github.com/NorthlandPositronics/Cogtrix/commit/36f1d35a1e359c56b0a68bf65f2928997eb78f0b))
- resolve BUG-116 — ApiConfirmationUI.render_prompt unblocks displaced callers ([e0ef573](https://github.com/NorthlandPositronics/Cogtrix/commit/e0ef57319d69e92024dc2f661d6c7ce28961d50c))
- resolve BUG-120 and BUG-126 in api/routes/sessions.py ([547ba15](https://github.com/NorthlandPositronics/Cogtrix/commit/547ba1535f49d5dbccd2a1649b60e54219a0c085))
- resolve BUG-122, BUG-113, and PERF-03 in session_bridge ([8080f65](https://github.com/NorthlandPositronics/Cogtrix/commit/8080f65347d935e0563f781c4ea1c8edd84e987e))

## [0.1.18](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.17...v0.1.18) (2026-03-05)

### Documentation

- holistic documentation audit — fix tool count, accuracy, and completeness ([c106408](https://github.com/NorthlandPositronics/Cogtrix/commit/c106408c4df4bf62cdbb100403ac824eed2f8390))

## [0.1.17](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.16...v0.1.17) (2026-03-05)

### Bug Fixes

- add missing `api` optional-dependency group to pyproject.toml ([a1a3af1](https://github.com/NorthlandPositronics/Cogtrix/commit/a1a3af1f43591582bb73bba11d262b36539df047))

## [0.1.16](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.15...v0.1.16) (2026-03-05)

### Features

- **api:** REST + WebSocket API layer ([d711932](https://github.com/NorthlandPositronics/Cogtrix/commit/d711932280889996d198f3f26a40430cb06b7d39))
- **api:** REST + WebSocket API layer with JWT auth, session management, and streaming agent turns ([7e6e4c2](https://github.com/NorthlandPositronics/Cogtrix/commit/7e6e4c2b083879739d7341d392bef930d229f29d))

### Bug Fixes

- exclude src/api from pyright — optional deps not in CI base install ([1927383](https://github.com/NorthlandPositronics/Cogtrix/commit/19273830863ef469b57adaf38e395dbd3c8a20d6))
- resolve CI failures — ruff B008/UP046 ignores, bandit B108, test import guards ([f4d6a5c](https://github.com/NorthlandPositronics/Cogtrix/commit/f4d6a5cc6477dde99f02041da4a79ee793328b79))
- skip API test files gracefully when fastapi is not installed ([05e8949](https://github.com/NorthlandPositronics/Cogtrix/commit/05e894912fdf52d9fa04f55dc2176ac16f9427c1))

## [0.1.15](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.14...v0.1.15) (2026-03-04)

### Bug Fixes

- resolve BUG-091 through BUG-099 in deferral system and adjacent files ([2406222](https://github.com/NorthlandPositronics/Cogtrix/commit/2406222e769c54b86ac1ec125def172f6d1b7ef1))
- resolve BUG-100 through BUG-104 in assistant subsystem ([efb9fd6](https://github.com/NorthlandPositronics/Cogtrix/commit/efb9fd6dbdb8d1c5c1a154fbec46a1f3d9ed2560))
- resolve BUG-105 through BUG-108 and BUG-094 partial fix ([8b69c61](https://github.com/NorthlandPositronics/Cogtrix/commit/8b69c617dcfd880479156af17111c50b4b5c6aad))
- resolve BUG-109 through BUG-112 and skip recovery on defer/suppress ([aca5f07](https://github.com/NorthlandPositronics/Cogtrix/commit/aca5f07049b78d59493245a02d9ca5aa48cc87c7))

## [0.1.14](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.13...v0.1.14) (2026-03-03)

### Features

- add defer_processing and suppress_reply tools for deferred message reasoning ([d910c1b](https://github.com/NorthlandPositronics/Cogtrix/commit/d910c1b006a8e24d09edc770a936f284d0d15e53))
- **assistant:** add queue_reply tool for sequential message delivery ([72e7267](https://github.com/NorthlandPositronics/Cogtrix/commit/72e72671bc3d427621009573920d324b0391a083))

### Bug Fixes

- apply round-6 audit sprint B fixes (BUG-083, BUG-084, PERF-1004, ARCH-040-10, ARCH-040-05, ARCH-040-04, ARCH-040-12, ARCH-040-06, ARCH-040-09, PERF-1007) ([0b4e9c2](https://github.com/NorthlandPositronics/Cogtrix/commit/0b4e9c2552df69ab35dbb1e122debdd9561ef7ea))
- **assistant:** apply queue_reply audit fixes (BUG-079, BUG-080, BUG-081, H1, H2) ([09d7ecf](https://github.com/NorthlandPositronics/Cogtrix/commit/09d7ecf29e8dfc497d04db668ea8ccdf4eba3c2b))
- **assistant:** apply round-6 sprint-A audit fixes ([75cb8b1](https://github.com/NorthlandPositronics/Cogtrix/commit/75cb8b1370f324e724272aa9cb7f1a571d0a0c7e))
- extract atomic_write_json utility and fix fd leaks (BUG-030, BUG-062, BUG-075) ([98b4595](https://github.com/NorthlandPositronics/Cogtrix/commit/98b4595691f6327d2aa273d7c62cdf28db4b6f1f))
- **guardrails:** remove broken ViolationTracker save debounce ([e382969](https://github.com/NorthlandPositronics/Cogtrix/commit/e38296969bdf8fde3efb39d6997e0ebe3ce448da))
- round-6 holistic audit — 18 findings across 14 files ([095e86a](https://github.com/NorthlandPositronics/Cogtrix/commit/095e86aceb12e1f971b36b9e1f87280c92888396))
- Sprint 1 — critical safety and correctness fixes (round 7) ([81e77fe](https://github.com/NorthlandPositronics/Cogtrix/commit/81e77feb309709f715aa96fb2bc0545f50bf20dd))
- sprint 3 — wizard template, SSRF guard, circuit breaker lock, MCP TOCTOU, LRU merge ([f861757](https://github.com/NorthlandPositronics/Cogtrix/commit/f8617570ab6926a74205ea2399863d04ab677ff3))
- sprint 4 — flush_all timer race, module-level thread pools for compression and tool execution ([c092e27](https://github.com/NorthlandPositronics/Cogtrix/commit/c092e27d41df0154b29406f23e3d0a1112d862c7))

### Documentation

- update CLAUDE.md for round-7 bug fixes across all 4 sprints ([7551967](https://github.com/NorthlandPositronics/Cogtrix/commit/7551967b0fac881f47122431313a7ff525768a8b))

## [0.1.13](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.12...v0.1.13) (2026-03-03)

### Bug Fixes

- apply Round 5 audit fixes (BUG-078/079/080/082, ARCH-039, PERF-901/902/907) ([2adf625](https://github.com/NorthlandPositronics/Cogtrix/commit/2adf625b65dbfe62a7dc2ccfd5b19bc7893af625))
- merge channel-specific config into WhatsApp/Telegram channel constructors ([840791a](https://github.com/NorthlandPositronics/Cogtrix/commit/840791af00dcd0bcbba34a74c0b907b778e82684))
- resolve 9 bugs and performance issues (BUG-074/075/076/077, PERF-802/804/806, ARCH-037-07/11) ([6291a65](https://github.com/NorthlandPositronics/Cogtrix/commit/6291a65d1aa44a245da3d36836ffcadca4dc7043))
- WAHA client-side filtering, factorial cap, rag atomic swap, and CI workflow ([712007c](https://github.com/NorthlandPositronics/Cogtrix/commit/712007c7411b449244c617a9e50aae9ad9534582))

## [0.1.12](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.11...v0.1.12) (2026-03-03)

### Features

- **assistant:** add filter_mode renames and blacklist delete/archive ([2eeae99](https://github.com/NorthlandPositronics/Cogtrix/commit/2eeae993c3fbca4298f83699e56d216c54846fc8))
- **assistant:** add message debounce buffer, edit_last_reply tool, and batch handling ([4f77f30](https://github.com/NorthlandPositronics/Cogtrix/commit/4f77f30fffe2fbf9b19c7668964ce6770d4bee25))
- **assistant:** add message editing, queue management tools, and bug fixes ([d571508](https://github.com/NorthlandPositronics/Cogtrix/commit/d571508588fd0209f94c48a744d1a1a3a45f57a6))
- **assistant:** add scheduler queue management tools and recipient tracking ([ef4a740](https://github.com/NorthlandPositronics/Cogtrix/commit/ef4a74090e9f5a1643e0569c4550ab47c7446469))
- **scheduler:** add chat_id and contact_name filters to list_scheduled_messages ([d27b9df](https://github.com/NorthlandPositronics/Cogtrix/commit/d27b9df80a1d35d4395ec5ce0ffb20e00aebea05))
- **whatsapp:** add two-phase polling tests and fix snapshot eviction order ([df3e588](https://github.com/NorthlandPositronics/Cogtrix/commit/df3e588b190bab6cebac8c7a797ddbeea1403c79))
- **whatsapp:** implement two-phase polling architecture ([a34422a](https://github.com/NorthlandPositronics/Cogtrix/commit/a34422a5d59a8d3796ead418d0c140e728b715c0))

### Bug Fixes

- apply Round 3 audit fixes (BUG-068/069/071/072, ARCH-035-01/03/13) ([16dcba1](https://github.com/NorthlandPositronics/Cogtrix/commit/16dcba10a36b4aea39472aaeaaed501f21f56c70))
- **assistant:** fix 5 polling and duration bugs (BUG-055 through BUG-059) ([552e0a3](https://github.com/NorthlandPositronics/Cogtrix/commit/552e0a30f3a26001cb2057f5f34c54dfe36daadb))
- **assistant:** remove early return in \_route_response so edit+schedule both fire ([0a7b1ea](https://github.com/NorthlandPositronics/Cogtrix/commit/0a7b1ead78460d3ee74d13e4dfcfd0d96991dacb))
- close leaked file descriptors and fix TOCTOU race in MCP loop creation ([cf775b3](https://github.com/NorthlandPositronics/Cogtrix/commit/cf775b3a9da21b111be55e67c2cfe202ba2368fc))
- implement ADR-0036 deferred audit fixes (6 items) ([3348baf](https://github.com/NorthlandPositronics/Cogtrix/commit/3348bafa40744b14ae41d5fad5f2886670e3da04))
- **scheduler:** add lock+idempotency to tool closures and coerce from_dict types ([254d411](https://github.com/NorthlandPositronics/Cogtrix/commit/254d4119725eb29f972ce182ff552d2d32ce1b28))
- **whatsapp:** fix 6 polling bugs (BUG-049 through BUG-054) ([893be0b](https://github.com/NorthlandPositronics/Cogtrix/commit/893be0be2ce560612405416c93ef05dadfaa04d3))
- **whatsapp:** rewrite polling to two-phase architecture, fix stale dedup cache ([df2acab](https://github.com/NorthlandPositronics/Cogtrix/commit/df2acab9d3f0915043e88bd30e34c608903bd1c4))

### Documentation

- holistic documentation revision syncing all docs with codebase ([e70576d](https://github.com/NorthlandPositronics/Cogtrix/commit/e70576da66ddb8e34abb55ce8108baf496d39460))

## [0.1.11](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.10...v0.1.11) (2026-03-02)

### Bug Fixes

- cache off-by-one and fd leaks in bound-cache, guardrails, knowledge, and json_store ([191c86f](https://github.com/NorthlandPositronics/Cogtrix/commit/191c86fd9853167a93e7c586c3ab943409ca8e90))
- deep-copy active_tools_list in run_execution_phase and cap compression fallback at \_FALLBACK_MAX_CHARS ([e89e501](https://github.com/NorthlandPositronics/Cogtrix/commit/e89e5016b8a03980f0e2c9b3738b380c10f0ccb1))
- eliminate orphaned tool-call chains, thread-safe slow-path counter, and LRU cache writeback ordering ([f4b72c0](https://github.com/NorthlandPositronics/Cogtrix/commit/f4b72c005c91259aff3a5e9a5a213b109d9a3ea8))
- Round 8 bug fixes, documentation revision, and assistant refactor ([38c7a2b](https://github.com/NorthlandPositronics/Cogtrix/commit/38c7a2b8137de6cd30d093b28c02cf6331f57c74))

### Documentation

- holistic documentation revision — fix tool count, memory window, changelog gaps ([dc83c57](https://github.com/NorthlandPositronics/Cogtrix/commit/dc83c57ec812d67be696aa9dc3e3fe81778affa4))

## [0.1.10](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.9...v0.1.10) (2026-03-02)

### Features

- **assistant:** add per-contact system prompts ([6c9d1c7](https://github.com/NorthlandPositronics/Cogtrix/commit/6c9d1c79bfcd0854fda434d5771b071a1db5989b))

### Bug Fixes

- **assistant:** contact prompt replaces system prompt, fix save_all/cleanup bugs, update docs ([6f68eb5](https://github.com/NorthlandPositronics/Cogtrix/commit/6f68eb59c84b58a914c8788d6bcd34f1a4928682))
- **assistant:** dynamic scheduler dispatch, crash-safe knowledge save, fix ViolationTracker.\_save lock ([992b6d8](https://github.com/NorthlandPositronics/Cogtrix/commit/992b6d8c365e837f266137c7074c01d9e432eb6a))
- **assistant:** fix [@lid](https://github.com/lid) contact prompt matching, persist stale expiry, respect excluded_tools ([791e703](https://github.com/NorthlandPositronics/Cogtrix/commit/791e7031f3c4e82116cb8a265aa1ed1421d7afae))
- **assistant:** fix four datamarking and PII bugs in handler.py ([8f1d259](https://github.com/NorthlandPositronics/Cogtrix/commit/8f1d25949a56ed7fd3e077182910bb03d27ec1e0))
- **assistant:** fix Telegram update replay, contact prompt spoofing, and rate limiter bypass ([b4bcb77](https://github.com/NorthlandPositronics/Cogtrix/commit/b4bcb77048be476535ece49002d582da99d9ada9))
- **assistant:** multi-round audit bug fixes and optimizations ([66bf868](https://github.com/NorthlandPositronics/Cogtrix/commit/66bf86816406ec6e5696792ee4ffd01136c09131))
- **assistant:** resolve remaining audit bugs and add scheduled reply prompt ([dab52e0](https://github.com/NorthlandPositronics/Cogtrix/commit/dab52e0c9da587fc2e193fb403fec6f34b0f7cae))
- correct documentation inaccuracies across config, README, and docs ([8bbb241](https://github.com/NorthlandPositronics/Cogtrix/commit/8bbb241a3af18af57cc2565723e766aedbf304f2))
- **scheduler:** recover in-flight messages on restart and add architectural review ([940cfb1](https://github.com/NorthlandPositronics/Cogtrix/commit/940cfb1d28fbec652ae3069c1177a7ca89504a74))
- **whatsapp:** add session start, LID resolution, and chats overview to client ([c54e8d6](https://github.com/NorthlandPositronics/Cogtrix/commit/c54e8d6d6f3b4480095657627fd419173d3da304))
- **whatsapp:** fix poll() chatId bug and add [@lid](https://github.com/lid) sender support ([a390e1c](https://github.com/NorthlandPositronics/Cogtrix/commit/a390e1cd5da01c54e8aecd8da61aa4b086b91b78))
- **orchestration:** fix active_tools_list mutation, compression fallback cap, LRU writeback ordering, and cache off-by-one (BUG-031..035) ([e89e501](https://github.com/NorthlandPositronics/Cogtrix/commit/e89e501), [f4b72c0](https://github.com/NorthlandPositronics/Cogtrix/commit/f4b72c0))
- **memory:** fix orphaned tool-chain cleanup, thread-safe slow-path counter, and fd leaks in json_store, guardrails, and knowledge (BUG-033,036,038..040) ([191c86f](https://github.com/NorthlandPositronics/Cogtrix/commit/191c86f))

### Performance Improvements

- **compression:** raise context compression threshold from 0.50 to 0.72 (PERF-001) ([f4b72c0](https://github.com/NorthlandPositronics/Cogtrix/commit/f4b72c0))

### Documentation

- document contact_prompts, schedule_reply, and response_timing in CONFIGURATION.md ([a24e762](https://github.com/NorthlandPositronics/Cogtrix/commit/a24e7622987d49c82b9d9b6725e22817f863cd79))
- document datamarking defense and scheduler recovery ([bca6437](https://github.com/NorthlandPositronics/Cogtrix/commit/bca6437594f3215f5ca5d86da83fe7d25344eff8))

## [0.1.9](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.8...v0.1.9) (2026-03-01)

### Features

- add user-constraint trust rule and milestone focus guidance to system prompt ([468e376](https://github.com/NorthlandPositronics/Cogtrix/commit/468e376b4c2110b7da93b066793237913aadaf2c))

### Bug Fixes

- add thread safety to ToolCallLogger, deduplicate tool-call key computation, and synchronize deep_think progress callback ([d54bfbd](https://github.com/NorthlandPositronics/Cogtrix/commit/d54bfbd1d6525d536c595b0b61c52151276ecbf7))
- address 5 low-severity bugs and perf issues (BUG-1829, BUG-1848, PERF-1101/1102/1103) ([3e4a583](https://github.com/NorthlandPositronics/Cogtrix/commit/3e4a583907121ddbfd756519fad31d67abf6a2a4))
- address three concurrent-safety and injection bugs ([24b0bc6](https://github.com/NorthlandPositronics/Cogtrix/commit/24b0bc643255c015b1b02d67c3e93d166f67dd6a))
- comprehensive security, thread-safety, and correctness audit (56 bugs fixed) ([7d6ad16](https://github.com/NorthlandPositronics/Cogtrix/commit/7d6ad167dac6d5f48113adac091b24b85e90ba7e))
- correct misleading log message, add CLI mutual-exclusion guard, and add hasattr guard in \_TokenAccumulator ([1694041](https://github.com/NorthlandPositronics/Cogtrix/commit/16940416f7f7be764c33761169c11f420e1cbd30))
- factorial DoS cap, JSON dot-path guard, exception types, session_state wiring ([b4e3c80](https://github.com/NorthlandPositronics/Cogtrix/commit/b4e3c805ebe39664dc151b6a13d00e58121562b3))
- guard against four High-severity bugs (BUG-1837..1840) ([40c2319](https://github.com/NorthlandPositronics/Cogtrix/commit/40c23197387cead25ca028f8651517ea568f6fac))
- handle plain host:badport in \_parse_ollama_address and early tmp_path assignment in setup wizard ([f3a65a2](https://github.com/NorthlandPositronics/Cogtrix/commit/f3a65a23b03e7e2d9cb250ba010c41ab699ec27c))
- MCP unsupported type warning, close_all iteration safety, handler approvals copy, guardrails test ([dbfa1e2](https://github.com/NorthlandPositronics/Cogtrix/commit/dbfa1e202c1837f4155c279356c441434fe045bc))
- **memory:** persist and restore mode-specific state across session restarts ([b5c07a3](https://github.com/NorthlandPositronics/Cogtrix/commit/b5c07a346390525a557ab45b813af20872cd2b0d))
- patch three confirmed bugs — SSRF in delegate URL re-fetch, intent false positives, and secret masking ([325cf99](https://github.com/NorthlandPositronics/Cogtrix/commit/325cf99397b9c2e618049e558398d44c6ae59541))
- path traversal guard in resolve_data_path and SSRF header blocking ([75999aa](https://github.com/NorthlandPositronics/Cogtrix/commit/75999aaf3c20d367c68e5d5518c5394fa7229e83))
- prevent ANSI corruption on non-TTY, fix spinner TOCTOU, and harden inline shell ([3cbd464](https://github.com/NorthlandPositronics/Cogtrix/commit/3cbd46423a1f49fae1667d59a5b76ca9ea26f6cf))
- resolve 4 medium-severity bugs (BUG-1852, BUG-1853, SEC-0802) ([9ac9b79](https://github.com/NorthlandPositronics/Cogtrix/commit/9ac9b79e7dc63a5d78adaa21fdee2ae4783147d9))
- resolve five medium-severity bugs (BUG-1828, BUG-1842, BUG-1844, BUG-1845, PERF-1100) ([1a453c9](https://github.com/NorthlandPositronics/Cogtrix/commit/1a453c9bb604e8598815763d20a8dc5f96867876))
- setup live-reload missing state rebuilds and ProviderConfig in-place mutation ([4cca9e4](https://github.com/NorthlandPositronics/Cogtrix/commit/4cca9e41c406cb2f9c98be86374218b24506b07a))
- warn on unsupported base_url in Google provider, atomic config swaps, broaden json_store exception handling ([34ffcb0](https://github.com/NorthlandPositronics/Cogtrix/commit/34ffcb051d6f3343119a8cbd311dce44a04d4956))

### Performance Improvements

- move hot-closure imports and regex literals to module scope ([d8a71f2](https://github.com/NorthlandPositronics/Cogtrix/commit/d8a71f211f1a6a432cdc4bb5dc59b2ba030385d0))

### Documentation

- update CLAUDE.md, AGENTS.md, README.md, and architecture docs for Rounds 6-10 ([86443a8](https://github.com/NorthlandPositronics/Cogtrix/commit/86443a80abc2f8d7fbe671073e6c1ffb948c625d))

## [0.1.8](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.7...v0.1.8) (2026-02-28)

### Features

- add top-level data_dir config option ([3cd3907](https://github.com/NorthlandPositronics/Cogtrix/commit/3cd3907d2427f13669d36e83d70d41b4b1be70c6))
- add top-level data_dir config option for all data storage paths ([ae13eb3](https://github.com/NorthlandPositronics/Cogtrix/commit/ae13eb308438f09274f4152d42bcd2bc85ba681e))
- **docker:** optimize Dockerfile with selective COPY and slim .dockerignore ([1c7311e](https://github.com/NorthlandPositronics/Cogtrix/commit/1c7311e81803409ef3586321d117f61bf2a6a420))
- **milestone:** add spinner context prefix and report_progress tool ([957066a](https://github.com/NorthlandPositronics/Cogtrix/commit/957066a60fa1e372cc315c491660042d066428e7))
- **milestone:** wire progress tracking into cogtrix.py (step 5) ([850bfb6](https://github.com/NorthlandPositronics/Cogtrix/commit/850bfb66dd8e3beaa50f3098db1d783c0c9ff6a5))
- **optimizer:** add Milestone/PromptPlan types and plan_milestones param ([79707f6](https://github.com/NorthlandPositronics/Cogtrix/commit/79707f698810cad0f35ee63a9771b7db44fde437))

### Bug Fixes

- resolve Round 26 audit bugs and performance issues ([2f8f6af](https://github.com/NorthlandPositronics/Cogtrix/commit/2f8f6af200b419a89bcf163d6d5a8a84500e61c7))

### Documentation

- add Round 26 audit reports (bugs, performance, architecture) ([c2c8a67](https://github.com/NorthlandPositronics/Cogtrix/commit/c2c8a67ae433b10ce6f118e93d63b7936818938c))

## [0.1.7](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.6...v0.1.7) (2026-02-28)

### Features

- add --allow-write-path flag to permit writes outside cwd ([dcd64ea](https://github.com/NorthlandPositronics/Cogtrix/commit/dcd64ea252e1ebbd83b31a7972dcd5af8f9a85e7))
- Rounds 19-25 — parallel execution, security hardening, and performance optimizations ([1510cb9](https://github.com/NorthlandPositronics/Cogtrix/commit/1510cb9fe488af7cbe3032d3aab027ae4b461b57))

### Bug Fixes

- **docker:** add missing extras for Anthropic, Google, MCP, and science ([4d6d33d](https://github.com/NorthlandPositronics/Cogtrix/commit/4d6d33df31a120733bed5f074d81c27c8635ba12))
- **file_ops:** allow read access to app install directory outside cwd ([9f362b4](https://github.com/NorthlandPositronics/Cogtrix/commit/9f362b4c558527bfdd0606e667a4c7b1b53cd57e))
- handle string allowed_write_paths, cap bound cache, copy config.available_tools ([aedab52](https://github.com/NorthlandPositronics/Cogtrix/commit/aedab5214730d24c6f5bddbc6ccc8b634f12a6aa))

### Performance Improvements

- connect MCP servers concurrently in connect_all() ([415ac13](https://github.com/NorthlandPositronics/Cogtrix/commit/415ac1362f16fe438a81bc5dfbc5de7437d9cd15))
- single-pass token estimation and parallel compression LLM calls ([1182068](https://github.com/NorthlandPositronics/Cogtrix/commit/1182068c77c1845007b35f3c020444676eb67a41))

### Documentation

- update documentation for Round 25 bug fixes and performance improvements ([66377ad](https://github.com/NorthlandPositronics/Cogtrix/commit/66377ade33372f78dbb696e9b2f22afb55b771f1))

## [0.1.6](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.5...v0.1.6) (2026-02-28)

### Features

- add optimizer feedback message and parallel tool execution ADR ([ff1986c](https://github.com/NorthlandPositronics/Cogtrix/commit/ff1986c6e349e10176d016e119533f71beb10061))
- implement parallel tool execution in process_tools node ([d016206](https://github.com/NorthlandPositronics/Cogtrix/commit/d016206610fe82e69110d4b48fed9c2a5c4b596e))

### Bug Fixes

- address three medium-severity bugs (BUG-1402, BUG-1403, BUG-1404) ([a9c97a9](https://github.com/NorthlandPositronics/Cogtrix/commit/a9c97a97efa58b0f755d5e183946983393e6fd6f))
- address three security/correctness bugs in runner and python_exec ([c0405fb](https://github.com/NorthlandPositronics/Cogtrix/commit/c0405fb6813e9454b1a191339fc6ca7d49bc55a0))
- break parallel futures loop on cancel and snapshot dict before iteration ([75e3016](https://github.com/NorthlandPositronics/Cogtrix/commit/75e3016d21459e90124b4f29b0ce6bf598b8a81c))
- bump version to 0.1.5 and fix release-please extra-files path ([8f74fa5](https://github.com/NorthlandPositronics/Cogtrix/commit/8f74fa57256999e3752541aa67e32330e9bd7a3f))
- close fd race, spinner dirty-terminal, and stderr fd leak ([8039cbd](https://github.com/NorthlandPositronics/Cogtrix/commit/8039cbd6f716c55736ba14aab104f15965a6cbad))
- derive no_confirm from self.\_session_state when set, falling back to True. ([1b1667e](https://github.com/NorthlandPositronics/Cogtrix/commit/1b1667e54a43293f86545e829382bb78e1c22c79))
- empty API key re-prompt, output cap module resolution, and wizard prompt injection ([9c89f0b](https://github.com/NorthlandPositronics/Cogtrix/commit/9c89f0b7854f474adf4354f54dc4068c449129b1))
- guard parallel block on cancel, fix UserCancelledRun in auto-expansion, pass parallel_tool_execution to MessageHandler, defer tool_list on cache-hit ([8ae422f](https://github.com/NorthlandPositronics/Cogtrix/commit/8ae422f72407ba57e02fc4a2c5c35eb2622f7a53))
- log tracebacks on agent/tool errors and defer old-LLM close until after swap ([574f95a](https://github.com/NorthlandPositronics/Cogtrix/commit/574f95ae3d52f05fd7f551f765c743d7bb57b298))
- **mcp:** upgrade connection cleanup log level and fix inter-server collision detection ([82f29b5](https://github.com/NorthlandPositronics/Cogtrix/commit/82f29b5d3bd3315fdcd131bdb0ea92cc1fb1619e))
- **orchestration:** stop serial-first loop on cancel and guard stale cache merge-back ([88dea96](https://github.com/NorthlandPositronics/Cogtrix/commit/88dea965eeaf35e30a1ad9579ab5668168ef022f))
- resolve 3 HIGH-severity bugs in approve toggle, event loop leak, and spinner race ([ca27ecb](https://github.com/NorthlandPositronics/Cogtrix/commit/ca27ecb4ab533e0a85b1322a0cd4f06e29c0f8a8))
- resolve Pyright type error in graph.py classification pass ([0efc0b5](https://github.com/NorthlandPositronics/Cogtrix/commit/0efc0b5cc0b365d9f1a681f3b06475b36a63f1ae))
- restore provider_config after rollback and respect no_confirm in MessageHandler ([1b1667e](https://github.com/NorthlandPositronics/Cogtrix/commit/1b1667e54a43293f86545e829382bb78e1c22c79))
- Round 18 audit fixes + holistic documentation revision ([f0fcf7a](https://github.com/NorthlandPositronics/Cogtrix/commit/f0fcf7a9704bd213073fd29145b40a8a0ca88951))
- Round 19-20 bug fixes, guardrails cleanup, and audit docs ([8097160](https://github.com/NorthlandPositronics/Cogtrix/commit/8097160e35cecc3e4b11d29e1d48a43c955b86b5))
- round 21 bug fixes — thread safety, cancel propagation, config wiring ([6aedc28](https://github.com/NorthlandPositronics/Cogtrix/commit/6aedc287609cf2eb47d7f4b49f871f048e4c76fd))
- round 22 bug fixes — cache isolation, cancel guards, session locks ([5459140](https://github.com/NorthlandPositronics/Cogtrix/commit/54591401ade342ab49c818db8ac957b20ff21e3f))
- **runner:** eliminate cache race condition in concurrent assistant mode ([0feb507](https://github.com/NorthlandPositronics/Cogtrix/commit/0feb507465c45d346eb30088dd88fad9e41659e4))
- serialize \_turn_count/\_section_ts in reasoning memory, fix falsy-string KeyError in compression, and bijective session ID sanitization ([80d379a](https://github.com/NorthlandPositronics/Cogtrix/commit/80d379a550aaa292482684b6d7f9bee7586edaf4))

### Performance Improvements

- move function-level imports to module level in runner.py ([dae4cd3](https://github.com/NorthlandPositronics/Cogtrix/commit/dae4cd39d27ce03b6625a0fb66a41b589bb53fb0))
- persist \_bound_cache and compression_cache across graph rebuilds ([b799c2f](https://github.com/NorthlandPositronics/Cogtrix/commit/b799c2f06ea4b79fc5137d660d71e8e573582f62))
- run optimize_prompt() concurrently with prepare_context() to reduce TTFT ([817cc9b](https://github.com/NorthlandPositronics/Cogtrix/commit/817cc9b5c495c8e13018ae852c2204cf05ca1e27))

### Documentation

- add ADRs 0015-0022, bug reports rounds 10-18, and update lockfile ([141bb8c](https://github.com/NorthlandPositronics/Cogtrix/commit/141bb8c36941cda26b0c04a7fc70f46ffd6aaf2c))
- add Round 18 bug hunt report ([858edd3](https://github.com/NorthlandPositronics/Cogtrix/commit/858edd30299c4da77f35e8858ae0b124f1836671))
- add Round 23 bug report and audit findings ([7f732f3](https://github.com/NorthlandPositronics/Cogtrix/commit/7f732f3e58dfaa00e70651b4374e654c3a87242c))
- add Round 24 audit reports (bugs, performance, architecture) ([bbcdcca](https://github.com/NorthlandPositronics/Cogtrix/commit/bbcdcca3fec43e1a4375f7434fbb9f397adf59f4))
- holistic documentation revision — fix accuracy drift and fill gaps ([af82852](https://github.com/NorthlandPositronics/Cogtrix/commit/af82852bfd159a1cd93164489eaf96f11f5ce1d0))
- update CLAUDE.md with parallel tool execution architecture ([d7653ec](https://github.com/NorthlandPositronics/Cogtrix/commit/d7653ec211e5f1a754fd15d9bcd5d675695f2387))

## [0.1.5](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.4...v0.1.5) (2026-02-27)

### Features

- add /setup slash command and MCP filesystem server to docker-compose ([67acf2d](https://github.com/NorthlandPositronics/Cogtrix/commit/67acf2d936794bd01bb8dbec2495a566b8ba191c))
- add cross-provider model resolution for /model command ([c63ee26](https://github.com/NorthlandPositronics/Cogtrix/commit/c63ee2662908a0a868994684101390a9cfbac2f5))
- add Escape key cancellation and Ctrl+C prompt re-editing ([1641afb](https://github.com/NorthlandPositronics/Cogtrix/commit/1641afb7496f4cbbc17338b6f28f4f2a2d4a4165))
- add search persistence guidance to system prompt and tool descriptions ([b505d2f](https://github.com/NorthlandPositronics/Cogtrix/commit/b505d2ffe39ca2184405966fb13b787d4c5ff12d))
- encourage http_get follow-up on search result URLs in system prompt and tool descriptions ([8555988](https://github.com/NorthlandPositronics/Cogtrix/commit/8555988c3b000839cc201f03c3711180106fbf1e))
- increase default RAG chunk_size from 1200 to 2000 characters ([eb1cc0a](https://github.com/NorthlandPositronics/Cogtrix/commit/eb1cc0abb3eea4f3a1f69e47abf17d7cfc87239a))
- **memory:** run summarization and embedding on a background daemon thread ([615084a](https://github.com/NorthlandPositronics/Cogtrix/commit/615084a0ace41dc3fb148d3fd50f8ac7f25d36d4))

### Bug Fixes

- add **post_init** validation to ProviderConfig (BUG-056) ([0fc34e5](https://github.com/NorthlandPositronics/Cogtrix/commit/0fc34e517210ac16a9c6c45076e6ec87ed2271a0))
- add 150ms warmup drain to prevent false Escape detection ([16b8384](https://github.com/NorthlandPositronics/Cogtrix/commit/16b838485d512d3a471dab7e6ae0dbca315b4c91))
- add bounded eviction to circuit breaker registry ([de42711](https://github.com/NorthlandPositronics/Cogtrix/commit/de42711241f3230d06480c73ce3ecb777d68dd03))
- add FAISS index lock and safe coercions for model numeric fields ([9a014ea](https://github.com/NorthlandPositronics/Cogtrix/commit/9a014ea4782704eff5b27399fe1f04771929a330))
- add injection-resistant delimiters to prompt optimizer ([13f1c7f](https://github.com/NorthlandPositronics/Cogtrix/commit/13f1c7fe6d8e256c5f1234775da0dc2afc3e4786))
- add warning for non-numeric IPv6 port, cap docs URL size, guard spinner stop race ([61b5783](https://github.com/NorthlandPositronics/Cogtrix/commit/61b5783ad2cbe8fc9790394f034001f022b176c3))
- address BUG-088/089/090/091 and PERF-009/011 bugs ([601621c](https://github.com/NorthlandPositronics/Cogtrix/commit/601621ccad6f28cbafa32549844718a462db78e0))
- address HIGH-severity bugs 083-087 ([dc0d65e](https://github.com/NorthlandPositronics/Cogtrix/commit/dc0d65ec53b524db8951c0bdcdf4826bb525f9ba))
- apply Round 17 audit fixes (ARCH-400..403) ([636d6b1](https://github.com/NorthlandPositronics/Cogtrix/commit/636d6b165d1aaad448fc4ca13ccd39032a0ab7eb))
- apply Round 17 audit fixes (PERF-300, PERF-301, ARCH-404, ARCH-405) ([1477d45](https://github.com/NorthlandPositronics/Cogtrix/commit/1477d45a1157ff2ffa91de319440d692f779dac7))
- apply Round 17 audit fixes BUG-702..705 ([3e6bb5b](https://github.com/NorthlandPositronics/Cogtrix/commit/3e6bb5b1aedeed08644d1bce9ddacbcb9edf55f2))
- **assistant:** create per-call SessionState to isolate concurrent chats ([80ed2fa](https://github.com/NorthlandPositronics/Cogtrix/commit/80ed2fa3c05ecb0c4c3af925d668c0c1ac9b94ae))
- **assistant:** hold lock during \_index_facts to prevent FAISS race condition ([33fe641](https://github.com/NorthlandPositronics/Cogtrix/commit/33fe641a8c9547b7476b03108cf61bd7d6724b4e))
- **assistant:** prevent shared dict/list mutation across concurrent sessions ([06431d0](https://github.com/NorthlandPositronics/Cogtrix/commit/06431d077551cd2a24b80af0ccaaee4c1bcbce19))
- BUG-042/051/044/047 — prompt injection, exception logging, MCP restart tools, registry fallback ([c9d8a9d](https://github.com/NorthlandPositronics/Cogtrix/commit/c9d8a9dd0ef1cec19688a98293b864aadeae1d4b))
- BUG-200/201/202 -- memory error prefixes, CGNAT SSRF, auth error ([67f92d4](https://github.com/NorthlandPositronics/Cogtrix/commit/67f92d443d5dad28cb46dc2861a0c9b6d054db63))
- close streaming response in \_follow_redirects before raising or redirecting ([0df0058](https://github.com/NorthlandPositronics/Cogtrix/commit/0df0058af973219a26b67cd30457dd40277e0603))
- **compression:** increase fallback truncation ratio from 50% to 75% ([0ad4f76](https://github.com/NorthlandPositronics/Cogtrix/commit/0ad4f76d0c34500478b4e3fb9b24f3a30b58fa71))
- **concurrency:** protect \_status_callback with a lock and hoist import time to module scope ([c88c76e](https://github.com/NorthlandPositronics/Cogtrix/commit/c88c76e719fd7eabe3b3dc1c2c0c374daa5bc320))
- correct IPv6 URL formatting and RAG vectordb_dir alignment ([c223e70](https://github.com/NorthlandPositronics/Cogtrix/commit/c223e703057c85dd360a118c2b07caae96779ffa))
- **deep_think:** shallow-copy LLM per thread in \_call_llm_parallel ([0b095e0](https://github.com/NorthlandPositronics/Cogtrix/commit/0b095e0157a80b570f1caa674ca6f94215e1109f))
- defer cbreak entry to monitor thread and remove prefill redisplay ([12339b0](https://github.com/NorthlandPositronics/Cogtrix/commit/12339b0707e6f704a6e256da9aba76cdb5270750))
- **delegate:** acquire \_circuit_breaker_lock around all check_availability() call sites ([d30c7d0](https://github.com/NorthlandPositronics/Cogtrix/commit/d30c7d0701a8ba5167938e6bc108e1cf9e1d8cb5))
- **delegate:** eliminate shared-object mutation race in run_research_delegate ([4e8e3f6](https://github.com/NorthlandPositronics/Cogtrix/commit/4e8e3f6a960441c8733f9f93d686a23481698f63))
- eliminate \_delegate_tools race condition, align ModelConfig errors, remove trivial executor ([e6b9d62](https://github.com/NorthlandPositronics/Cogtrix/commit/e6b9d625264a96ef266e9f2c40906e405b15f0ef))
- eliminate circuit breaker check_availability() race condition ([9233648](https://github.com/NorthlandPositronics/Cogtrix/commit/9233648dda13fffcc9817dc5f9ab457c20419e17))
- eliminate DNS rebinding TOCTOU, dead code, prompt lies, and circuit-breaker gaps ([74ef3ff](https://github.com/NorthlandPositronics/Cogtrix/commit/74ef3ff490ecb0a80622cd701857f95476599c32))
- eliminate race conditions in python_exec and http_request ([a71aa60](https://github.com/NorthlandPositronics/Cogtrix/commit/a71aa60821f1641562cb96d511d82b5230556c5c))
- eliminate thread-unsafe global fallback in python_exec session routing ([29149da](https://github.com/NorthlandPositronics/Cogtrix/commit/29149da46259f09f5b0ff522f0c5d0a0f57a41ec))
- ensure spinner is always resumed and make circuit breaker thread-safe ([d58605d](https://github.com/NorthlandPositronics/Cogtrix/commit/d58605da74019311d96db2d51676ee7adc6dde6e))
- **escape-monitor:** fix 5 bugs in warmup, stop, drain, and error handling ([0ce4ca8](https://github.com/NorthlandPositronics/Cogtrix/commit/0ce4ca8d484bf7b4eaf32927828d5cf764cc9116))
- exclude execute_python from assistant mode and scrub secrets in \_log ([5877a23](https://github.com/NorthlandPositronics/Cogtrix/commit/5877a23054af411396ba5bdb15aeff380cf9bd52))
- **file_ops:** block absolute path writes outside cwd (BUG-006) ([844b605](https://github.com/NorthlandPositronics/Cogtrix/commit/844b605df16fe28a64d89120efa95ad2e35a0a8b))
- **file_ops:** block absolute paths outside cwd in \_validate_path ([db11fb7](https://github.com/NorthlandPositronics/Cogtrix/commit/db11fb70703d2759a88b44d55371da2fcaa3e40b))
- **file_ops:** eliminate TOCTOU race in read_file by removing pre-checks ([6b96d3f](https://github.com/NorthlandPositronics/Cogtrix/commit/6b96d3f2a9a952226adbd5084082302bee06df4d))
- **file_ops:** remove unreachable is_write cwd check in \_validate_path() ([4b691d2](https://github.com/NorthlandPositronics/Cogtrix/commit/4b691d2b4f2698313156d0803316bc47d997c051))
- fix nullable anyOf/oneOf in MCP schema and turn_count in wrong method ([445ab56](https://github.com/NorthlandPositronics/Cogtrix/commit/445ab56a05ee2d8494c92c6eee34e10aafa52167))
- flush stale stdin bytes to prevent false Escape detection on first prompt ([f4a2087](https://github.com/NorthlandPositronics/Cogtrix/commit/f4a2087014f18086b4900ffa62d2a40a3d30cb7c))
- guard int/float coercions against non-numeric strings and fix phases.py provider bypass ([3720a45](https://github.com/NorthlandPositronics/Cogtrix/commit/3720a45226813f196eda0b99e82d702403534143))
- **guardrails:** eliminate TOCTOU in ChatRateLimiter and fix leet-score false positives on numeric tokens ([4da74b9](https://github.com/NorthlandPositronics/Cogtrix/commit/4da74b9b712c50083210e50cac5c8ccdb2368b84))
- **guardrails:** wire ViolationTracker persist_path and release lock before disk write ([f98f286](https://github.com/NorthlandPositronics/Cogtrix/commit/f98f28670caea2c827267abd703efb4cdcbaf6c9))
- handle nested triple-backticks in \_extract_yaml greedy fallback ([0e636e1](https://github.com/NorthlandPositronics/Cogtrix/commit/0e636e166fc2166e597aeda468fefe20c4c1fd77))
- handle ToolMessage in on_tool_end and clean stale tool_lookup entry ([311f31c](https://github.com/NorthlandPositronics/Cogtrix/commit/311f31cdede474da62fde713d7be3484899f5660))
- **handler:** record rate-limit before agent run and extract knowledge before sanitize ([9ddbb74](https://github.com/NorthlandPositronics/Cogtrix/commit/9ddbb74918ad07d3043e09d8b5e3ff90448011e7))
- harden security in setup_wizard, whatsapp channel, and python_exec sandbox ([b6e8d97](https://github.com/NorthlandPositronics/Cogtrix/commit/b6e8d976b0181f944224147d0b14bf4f8969ae91))
- **http:** eliminate DNS rebinding TOCTOU in http_request tool (BUG-074) ([90003db](https://github.com/NorthlandPositronics/Cogtrix/commit/90003dbc5716aee3312f620b135757144ccd1a1d))
- **http:** evict stale entries from \_recent_failures in \_record_failure ([c22f4b3](https://github.com/NorthlandPositronics/Cogtrix/commit/c22f4b3292797616c67d197225fdfe56a6114365))
- **http:** handle iter_content exceptions gracefully in \_read_bounded_response ([3527c1f](https://github.com/NorthlandPositronics/Cogtrix/commit/3527c1f12668912c2fc5cf81dc705913491c449c))
- **http:** stream responses to avoid buffering large bodies in memory ([65fbaca](https://github.com/NorthlandPositronics/Cogtrix/commit/65fbaca3da5da36e5f88ed0ea495d8a682c7ce41))
- implement P2 correctness fixes F2, F9 (partial) ([4549b9b](https://github.com/NorthlandPositronics/Cogtrix/commit/4549b9bb0491fdc27232f032069daa517fe26dbd))
- **intent:** widen explain-verb proximity guard threshold from 30 to 50 ([35a1827](https://github.com/NorthlandPositronics/Cogtrix/commit/35a1827a6bb86802964af9bb1db771eb79ed41a9))
- lazy history paths and monotonic clock in ViolationTracker ([e61ce79](https://github.com/NorthlandPositronics/Cogtrix/commit/e61ce79d39a464798f202984c6f491da7791f3d2))
- **logging:** scrub secrets from tool inputs in on_tool_start ([71b6672](https://github.com/NorthlandPositronics/Cogtrix/commit/71b667238671afa11b3bfb079f64fdc55cf265de))
- **logging:** scrub secrets from tool_args before logging LLM_TOOL_CALL ([20f5f23](https://github.com/NorthlandPositronics/Cogtrix/commit/20f5f2386e7f8eae48d2eec006a9adcbdc3bd21f))
- **mcp:** handle complex JSON Schema types and builtin tool name collisions ([d8b446e](https://github.com/NorthlandPositronics/Cogtrix/commit/d8b446ed265b3a57e523c2736ab94d25e8f43903))
- **memory:** atomic write in save_history() to prevent corrupt JSON on crash ([d14844f](https://github.com/NorthlandPositronics/Cogtrix/commit/d14844fcbcb56a36fc36cb6d51e18afab6230991))
- **memory:** guard \_save_hybrid_meta reads under lock, atomic write, shallow-copy batch ([db09700](https://github.com/NorthlandPositronics/Cogtrix/commit/db09700b669022b147ea557dbf29c50555783adf))
- move deny_all/denials check inside confirmation_lock to eliminate TOCTOU race ([c81850f](https://github.com/NorthlandPositronics/Cogtrix/commit/c81850f54019eb45f244ded75150b8db80740947))
- move pause_spinner inside try block and lock circuit breaker record calls ([0823571](https://github.com/NorthlandPositronics/Cogtrix/commit/08235710fe6b3e35f78ad9700b0fc3cca856ff8e))
- parse num_ctx/temperature from providers section and add missing env var handlers ([887971d](https://github.com/NorthlandPositronics/Cogtrix/commit/887971d04acdb6993934e27f7fe2bbab37979b8c))
- pass AIMessage to \_detect_tool_request so explicit tool loading works ([e900b50](https://github.com/NorthlandPositronics/Cogtrix/commit/e900b502921b94143ff369d1bf1a642069dd109c))
- pass session_state to run_agent/run_execution_phase and split \_enter_cbreak try blocks ([8a2b0fb](https://github.com/NorthlandPositronics/Cogtrix/commit/8a2b0fb03a825411bcc16f8d528af80ad0b801e0))
- persist memory on exit and update hybrid test for background threading ([afaf00f](https://github.com/NorthlandPositronics/Cogtrix/commit/afaf00f4918ed05e9dc98abd1e908d981261c384))
- persist ViolationTracker blacklist state across restarts ([051f262](https://github.com/NorthlandPositronics/Cogtrix/commit/051f26278681c111f1890698bd19ad8313f384a8))
- **phases:** make extract_turn_messages robust with isinstance and boundary anchor ([13ecdc9](https://github.com/NorthlandPositronics/Cogtrix/commit/13ecdc943873152060b5e95abd71401f4a8da83e))
- preserve wizard model key and rebuild compression LLM on switch ([e0a0577](https://github.com/NorthlandPositronics/Cogtrix/commit/e0a05775c9d0f366563b548a8888546d62588102))
- prevent double optimizer invocation on /o force-optimize command ([6f18b0c](https://github.com/NorthlandPositronics/Cogtrix/commit/6f18b0cbf43f6179f66219eeaf0ca2a250b99b4a))
- **prompt:** sanitize delimiter strings in optimizer to prevent injection ([cb16850](https://github.com/NorthlandPositronics/Cogtrix/commit/cb168506b7cd883eacde6900db8a5c4f95a85215))
- propagate UserCancelledRun and show spinner during optimize_prompt ([b97e7fb](https://github.com/NorthlandPositronics/Cogtrix/commit/b97e7fb0c7e22b8bfbe550954609f233ee244680))
- Pydantic v1/v2 copy compat and RAG subdir recursion (BUG-049, BUG-050) ([2f63861](https://github.com/NorthlandPositronics/Cogtrix/commit/2f638612fa71d40eafeadc3fb94122cd5848727e))
- **python_exec:** close sandbox escape via type.**dict** descriptor chain ([0aa183e](https://github.com/NorthlandPositronics/Cogtrix/commit/0aa183e3acd6c7ad8048ef06ffabd1ce87ff2811))
- **python_exec:** replace substring module check with AST imports and add LRU eviction ([01f87fc](https://github.com/NorthlandPositronics/Cogtrix/commit/01f87fc31f8496a7d86dfff447713cff0d242cab))
- **registry:** remove single-schema fallback in fallback tool discovery ([f6fee62](https://github.com/NorthlandPositronics/Cogtrix/commit/f6fee62a7a128043de16159043aa3771e3f7ae52))
- remove reverse import in handler.py and sync AgentRunner Protocol ([708f440](https://github.com/NorthlandPositronics/Cogtrix/commit/708f440baba2b68391588a6c1e1744f6a0770101))
- reset \_deny_all in run_single_prompt and guard \_compress_one exceptions ([9c8738d](https://github.com/NorthlandPositronics/Cogtrix/commit/9c8738d301366a0ba34b216f821296d9090de04e))
- resolve 5 P2/P3 bugs across orchestration, config, and CLI ([98ca0e5](https://github.com/NorthlandPositronics/Cogtrix/commit/98ca0e5b75ce9f5ecc252c4853cac5e1e7ea7b51))
- resolve circular imports in src/orchestration/phases.py ([5845590](https://github.com/NorthlandPositronics/Cogtrix/commit/5845590a23dbaaf01f8f0f61c83f2569185d8c16))
- resolve four bugs across mcp_client, deep_think, memory, and poller ([158124b](https://github.com/NorthlandPositronics/Cogtrix/commit/158124bfbc865cf5672050c5e1f950437e0d5e48))
- resolve graph.py dependency issues after module extraction ([ec229a4](https://github.com/NorthlandPositronics/Cogtrix/commit/ec229a4665340938934bf123e86bcb2afa9e1b28))
- resolve spinner deadlock and WEB_TOOL_NAMES divergence ([e950613](https://github.com/NorthlandPositronics/Cogtrix/commit/e950613d428e9a33ed9c7b80cba3ae89c0f97abf))
- scan only current iteration result_msgs in \_detect_tool_request ([e57c196](https://github.com/NorthlandPositronics/Cogtrix/commit/e57c196ba789bfda2a6defd466316a26e0d414d7))
- **security:** add Unicode bidi isolate codepoints and fix circuit breaker lock race ([1443b1d](https://github.com/NorthlandPositronics/Cogtrix/commit/1443b1d2297df5092a8b40512b7a50175e35e16a))
- **security:** atomic writes for knowledge/violation stores; sanitize SDK errors in agent error formatter ([7aaab51](https://github.com/NorthlandPositronics/Cogtrix/commit/7aaab511f7588577e080acd0c5efb67767429f90))
- **security:** block SSRF via redirect in http_get and http_post ([cf7499c](https://github.com/NorthlandPositronics/Cogtrix/commit/cf7499c09e4e88469e2334bec3fda1766be6d665))
- **security:** close sandbox escape via runtime getattr/setattr (BUG-016) ([601e7f9](https://github.com/NorthlandPositronics/Cogtrix/commit/601e7f91dfae15b3c588b71565227f27eb4d5888))
- **security:** correct shell tool exclusion name and add homoglyph normalization to guardrails ([943c894](https://github.com/NorthlandPositronics/Cogtrix/commit/943c8949d11bfe2a50a8f7db55f8617016b3e02f))
- **security:** normalize paths before prefix-checking in ToolCallGuard and recall.py ([98a3af0](https://github.com/NorthlandPositronics/Cogtrix/commit/98a3af0a7907cd20fe94541d4a9cc7cefdfdf8fc))
- **security:** replace string-based SSRF checks with ipaddress+socket validation ([dd7813a](https://github.com/NorthlandPositronics/Cogtrix/commit/dd7813a81999c90d6471715718772b349d50de19))
- **security:** scrub LLM output, add xai- key prefix, and fix sandbox hasattr ([c0bba06](https://github.com/NorthlandPositronics/Cogtrix/commit/c0bba0622da1178de91932e5df62cac0a60b5471))
- send response before memory update and add RLock to SessionVectorStore ([1df44ac](https://github.com/NorthlandPositronics/Cogtrix/commit/1df44ac946554d8f125cc3f78df92b7b732d72bd))
- serialize ViolationTracker disk writes under lock and replace O(n²) list pops with O(n) slice ([1e025cd](https://github.com/NorthlandPositronics/Cogtrix/commit/1e025cdaa9a2beb4232c6822b3b383cb0672e995))
- **shell:** return accurate message when command fails with no output ([e187979](https://github.com/NorthlandPositronics/Cogtrix/commit/e187979eab7c03f12e5429e29aa15e54748adb0e))
- split auto_expansion_count, distinguish active/unknown tool errors, add fuzzy rename guidance ([4da7445](https://github.com/NorthlandPositronics/Cogtrix/commit/4da7445a30d7ed3bbf2e80bb8927aec8157cef6d))
- **ssrf:** block link-local, RFC 6598, IPv6 ULA, and IPv4-mapped loopback in \_validate_url ([ed7778b](https://github.com/NorthlandPositronics/Cogtrix/commit/ed7778b417bfba78e8474af856c7c3156575856c))
- stop LLM echoing timestamps, fix pipe table rendering, dim shell output ([686a230](https://github.com/NorthlandPositronics/Cogtrix/commit/686a2304e0e5858f3142dddddfe4c53d6073cea4))
- stop mutating shared ProviderConfig during model resolution ([c180c1f](https://github.com/NorthlandPositronics/Cogtrix/commit/c180c1fd55c10cbf49f9e9e0cde677460fab9974))
- stop spinner race, re-raise UserCancelledRun, add exit hint ([e8abd26](https://github.com/NorthlandPositronics/Cogtrix/commit/e8abd26bb7e977fcce6463cb755ba1ebc6abdd89))
- **thread-safety:** atomic config swap and stderr lock for concurrent tool calls ([ae68ac8](https://github.com/NorthlandPositronics/Cogtrix/commit/ae68ac8928d9a746433ad594fb4cb493711458ab))
- **threadpool:** replace context-manager executors to prevent timeout blocking ([7fff03c](https://github.com/NorthlandPositronics/Cogtrix/commit/7fff03cf2bdf50592b7de4d65a6d651a48e4162f))
- three bug fixes — no_confirm bypass, list-form ToolMessage content, lock scope ([497fdbe](https://github.com/NorthlandPositronics/Cogtrix/commit/497fdbefdd0c197dec452803c66724f6e501b5dc))
- three correctness bugs in check_config, execution phase, and output cap ([b6b7413](https://github.com/NorthlandPositronics/Cogtrix/commit/b6b74132e26918bb39024548a26468747fe42f95))
- validate negative integer config fields and normalize provider type case ([3227cdd](https://github.com/NorthlandPositronics/Cogtrix/commit/3227cddfb64ed49cf8ed0e9d23efdb7394f24a8b))

### Performance Improvements

- cache bind_tools() result in call_model to avoid redundant schema rebuilds ([53cfe9b](https://github.com/NorthlandPositronics/Cogtrix/commit/53cfe9bd5b6271825ff427a1b8a94628eeee43a6))
- **deep_think:** remove duplicate ISOLATION WARNING from tool description ([0faa38b](https://github.com/NorthlandPositronics/Cogtrix/commit/0faa38b9792046d2f0a005288b8d1061135404f6))
- implement P2 TTFT optimizations (F1, F3, F8) ([a4ac8d9](https://github.com/NorthlandPositronics/Cogtrix/commit/a4ac8d969c45a5535ac8481d791fa5fe1d27b7c4))
- lift tool_lookup rebuild and skip blocking join in save() ([9f9c24b](https://github.com/NorthlandPositronics/Cogtrix/commit/9f9c24ba3d09cbb4978dc0194e566217c8f3723e))
- **memory:** gate stale reasoning prefix sections to reduce TTFT (F5) ([9f78f11](https://github.com/NorthlandPositronics/Cogtrix/commit/9f78f11eecc6cc5373081df6b08fd893f75cf693))
- **optimizer:** raise length gate from 150 to 400 chars, action-verb skip to 600 ([447f746](https://github.com/NorthlandPositronics/Cogtrix/commit/447f74690e01ee328eba571c4b3196485e61d27a))
- remove blocking \_wait_for_background from prepare_context, cap escape drain loop ([d9de96f](https://github.com/NorthlandPositronics/Cogtrix/commit/d9de96ffdf06d592764306eb641a3ed30e2b0655))
- remove duplicate search guidance, hoist time import, cache SystemMessage ([1b8db52](https://github.com/NorthlandPositronics/Cogtrix/commit/1b8db52f9467473759f5700d2ce34b836a6a39fd))
- **request_tools:** remove tool name list from description to reduce token usage ([665b36b](https://github.com/NorthlandPositronics/Cogtrix/commit/665b36b6805591164ae4a2755c1707db53c3d449))
- tune memory and compression thresholds (P2 batch) ([3487a46](https://github.com/NorthlandPositronics/Cogtrix/commit/3487a4613868a8921dbf00c3c4e852edddf42b2a))

### Documentation

- add ADR 007 — unify tool safety architecture decision record ([91ddcdd](https://github.com/NorthlandPositronics/Cogtrix/commit/91ddcdd10dfe4e63fd9bef55dd50d4cae2cd4b4a))
- add architecture refactoring plan ([820169c](https://github.com/NorthlandPositronics/Cogtrix/commit/820169c9ecf94149f21fda894fd0b79d2901da16))
- add post-refactor bug sweep and AI interaction audit reports ([645be13](https://github.com/NorthlandPositronics/Cogtrix/commit/645be138960d89bbe8d0af246eb2dc2237860182))
- add Round 3 audit ADRs and findings reports ([a44f0c9](https://github.com/NorthlandPositronics/Cogtrix/commit/a44f0c97d6c6e2a741e54680187b1d2a81dccfa4))
- fix tool count inconsistency — standardise to 51 across all docs ([85fd20a](https://github.com/NorthlandPositronics/Cogtrix/commit/85fd20a23e4d1288f1c18653ee9fdcb90873cda3))
- holistic documentation revision + P4 bug fixes + Round 6 audit prep ([28ef1ee](https://github.com/NorthlandPositronics/Cogtrix/commit/28ef1ee9b0541a1b83cd50b552c016c56aee958d))
- update CLAUDE.md and bug reports to reflect all ProjectForge audit fixes ([8380e08](https://github.com/NorthlandPositronics/Cogtrix/commit/8380e08ae606b5b5b9b69504cf273eaaaead35a1))
- update CLAUDE.md for Round 3 fixes and add verification report ([1f7409d](https://github.com/NorthlandPositronics/Cogtrix/commit/1f7409d6492260970c8791bd1c5d365ecffb2b9e))
- update CLAUDE.md to reflect new module structure ([aad17a6](https://github.com/NorthlandPositronics/Cogtrix/commit/aad17a6308a4523731bd3f732ab83c087f964ba9))
- update documentation to reflect bug fix changes ([b7135bc](https://github.com/NorthlandPositronics/Cogtrix/commit/b7135bc0c0d1e2996fdb7fdbf6a3e54ae527a9de))

## [0.1.4](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.3...v0.1.4) (2026-02-22)

### Bug Fixes

- **ci:** align release workflow with CI pipeline ([4a8e783](https://github.com/NorthlandPositronics/Cogtrix/commit/4a8e783861b1f602f2b578657e6bde17cf6dafa1))
- **ci:** exclude integration tests and set bandit threshold in release workflow ([a5f7af3](https://github.com/NorthlandPositronics/Cogtrix/commit/a5f7af30a01c279c9d9d1f8644483e1258f6b434))

## [0.1.3](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.2...v0.1.3) (2026-02-22)

### Features

- activity indicator (spinner) during LLM processing ([178a1cc](https://github.com/NorthlandPositronics/Cogtrix/commit/178a1ccfb51a900a6d056f3429550c01a813e794))
- add --setup, --setup-docs, --setup-output CLI flags and dispatch handler ([07da36f](https://github.com/NorthlandPositronics/Cogtrix/commit/07da36f551c84fc48abe8fb5d455bd4f8c6d7756))
- add /optimizer &lt;prompt&gt; to force-optimize and run a prompt ([2210421](https://github.com/NorthlandPositronics/Cogtrix/commit/22104217cb9f2599cb5efb5b4e4707e7301d4cfc))
- add /optimizer command, rename /noconfirm to /approve, add aliases ([b4d221d](https://github.com/NorthlandPositronics/Cogtrix/commit/b4d221d9fd2d66f169fcc16b726e484631c05a19))
- add /tools enable/disable subcommands and on-demand tool status ([256e486](https://github.com/NorthlandPositronics/Cogtrix/commit/256e486e2d7fffdabe93c33903f22003201951de))
- add /tools load subcommand and [loaded] status tag ([ac9058e](https://github.com/NorthlandPositronics/Cogtrix/commit/ac9058edc472fc20befa9f26909ca48c603201af))
- add agent message-handling workflow integration tests ([b757ff6](https://github.com/NorthlandPositronics/Cogtrix/commit/b757ff69eb9dde2ec0fdc6dac3ec39ad71214f45))
- add Anthropic/Google provider extras, expand Docker multi-provider setup ([51e517b](https://github.com/NorthlandPositronics/Cogtrix/commit/51e517b8a1fe50343d8e4e802403efe5070cc88c))
- add CLI flags and file-ref config for assistant system prompt ([1cb1487](https://github.com/NorthlandPositronics/Cogtrix/commit/1cb1487d11fa3041e8d1b5489499ead85f0c1f89))
- add colored response stats, fix TokenAccumulator, show system prompt in /info ([ead03ca](https://github.com/NorthlandPositronics/Cogtrix/commit/ead03caa6b77712314970b22b3c656a2a5e5fc6a))
- add colorized --help page with structured argument groups ([953240f](https://github.com/NorthlandPositronics/Cogtrix/commit/953240f624af486454adb3a5b8da23012918c52c))
- add dedicated compression model support via context_compression.model config ([29f3670](https://github.com/NorthlandPositronics/Cogtrix/commit/29f3670b0d09d6ecc33339fb64161606a4ca93d2))
- add delegation visibility, /delegate command, and auto-delegation ([a85ef95](https://github.com/NorthlandPositronics/Cogtrix/commit/a85ef95b418815464f382006795a9ae3dbf4b568))
- add EncodingDetectionGuard and ToolCallGuard to guardrail pipeline ([4445a6d](https://github.com/NorthlandPositronics/Cogtrix/commit/4445a6d80f21fa1ba664c30d24ecb03cdbe46ad8))
- add hidden /system_prompt command, show only prompt size in /info ([253a902](https://github.com/NorthlandPositronics/Cogtrix/commit/253a902f62bcbdb33d806c2883a8c87a90c359d7))
- add in-loop context compression for old ToolMessages ([dc3ba95](https://github.com/NorthlandPositronics/Cogtrix/commit/dc3ba956d1787ad4160a31590483ec9979be8c49))
- add inline shell commands, bright prompt, history resilience ([cb49340](https://github.com/NorthlandPositronics/Cogtrix/commit/cb493400c8615acbeba4b0292aed74b3d304d498))
- add MCP client manager module ([fc70705](https://github.com/NorthlandPositronics/Cogtrix/commit/fc7070511b9e9102865141734fb32c4d429ea5b7))
- add MCP config, registry helpers, optional dep, and documentation ([4e7d815](https://github.com/NorthlandPositronics/Cogtrix/commit/4e7d815faa30f2b2a09427871b1335cae5e35ae8))
- add prompt optimizer, StateGraph stream recovery, and tool instructions ([7aaed09](https://github.com/NorthlandPositronics/Cogtrix/commit/7aaed099dd318c4779c707d8a84d9dc78a9c2bba))
- add provider registry, auto-launch wizard, and fix assistant guardrails ([49db710](https://github.com/NorthlandPositronics/Cogtrix/commit/49db710d9bf9e48686f2b5253882d75e1d5431a2))
- add SharedKnowledgeStore for cross-chat fact extraction and recall ([2117a7b](https://github.com/NorthlandPositronics/Cogtrix/commit/2117a7beed83f8acd0c94a4f49171ee1b2d2ec77))
- add Telegram messaging tool, WhatsApp guide, and Docker Compose ([1a3be08](https://github.com/NorthlandPositronics/Cogtrix/commit/1a3be0859ff66c4f1c1fa07c1d45c1f8c7fd2959))
- add tool_call_guard callback to \_build_agent_graph and run_agent ([b6ec413](https://github.com/NorthlandPositronics/Cogtrix/commit/b6ec41318dfd02682bf19f84894b67b087934220))
- add UTC timestamps to message history and harden type safety ([6425276](https://github.com/NorthlandPositronics/Cogtrix/commit/642527640093a62fb80576eabb001f8543025077))
- add ViolationTracker and auto-blacklist to GuardrailPipeline ([16bbb3a](https://github.com/NorthlandPositronics/Cogtrix/commit/16bbb3a538c941457003f8158bd61148d274ca0f))
- add WhatsApp messaging tool and fix bugs in contact filtering ([a1d8e61](https://github.com/NorthlandPositronics/Cogtrix/commit/a1d8e61ae24d2866e282318c0b161da417b20bd1))
- allow model to release tools via request_tools meta-tool ([164623a](https://github.com/NorthlandPositronics/Cogtrix/commit/164623adab4a23434a3be8bd54ff53644d894bf2))
- auto version bump and release on merge to main ([fb2b89f](https://github.com/NorthlandPositronics/Cogtrix/commit/fb2b89f0e9bac86ccb1b198ab99293af978877bc))
- default to Ollama for zero-config out-of-box experience ([4c942af](https://github.com/NorthlandPositronics/Cogtrix/commit/4c942af28c391c57555881357a32839854e30780))
- default to Ollama for zero-config out-of-box experience ([98c2c7b](https://github.com/NorthlandPositronics/Cogtrix/commit/98c2c7b03575c3f031cfec6f8d5e1c6a67b73706))
- display response stats (elapsed time and token usage) after agent replies ([f4a25d6](https://github.com/NorthlandPositronics/Cogtrix/commit/f4a25d629fc4353136db8ed6de01986dd70a72fa))
- double spinner phrases, gradient color, trailing space ([f8aa7b5](https://github.com/NorthlandPositronics/Cogtrix/commit/f8aa7b5693bcc86bbe6bd2de26b0d72fa548575d))
- enhance setup wizard with Rich rendering, spinner, API key reuse, and tests ([c14ae85](https://github.com/NorthlandPositronics/Cogtrix/commit/c14ae8550d6bd497bc2e0b61bdccc339cedc8d72))
- expand spinner to 80 phrases with humorous tech messages ([c49cb79](https://github.com/NorthlandPositronics/Cogtrix/commit/c49cb794e47ce18a48080b514b63cb02081cf7eb))
- expand task classification to 23 categories and fix force deep_think override ([e71204b](https://github.com/NorthlandPositronics/Cogtrix/commit/e71204b21621280e807b74494b3018527e2ee2c2))
- expand tool confirmation prompt with disable/deny-all/cancel options ([65f787a](https://github.com/NorthlandPositronics/Cogtrix/commit/65f787a3cbcddc13f47a73a8c461ba332ac369b5))
- hybrid /think pipeline with category-aware prompts ([eede19b](https://github.com/NorthlandPositronics/Cogtrix/commit/eede19b32062c1e887e1ec27bd95a8a1085f4f0e))
- implement hybrid memory (sliding window + summary + vector recall) ([12835af](https://github.com/NorthlandPositronics/Cogtrix/commit/12835af3aec15960b46636a4b3138b26aa4e5b71))
- implement Sprint 1 of assistant mode (WhatsApp/Telegram daemon) ([a971ac5](https://github.com/NorthlandPositronics/Cogtrix/commit/a971ac5075cd9d81d12386805cc5445b970195be))
- increase working memory for code and reasoning modes ([dc44c66](https://github.com/NorthlandPositronics/Cogtrix/commit/dc44c66bf0b4a67747150f4026055bdf76c779f0))
- integrate MCP server support into CLI and slash commands ([1ab64f6](https://github.com/NorthlandPositronics/Cogtrix/commit/1ab64f6fcf49d132ed90821ce5e4063222002a92))
- move --setup to early-exit position and add auto-launch wizard ([ac348f0](https://github.com/NorthlandPositronics/Cogtrix/commit/ac348f01d1eb4d92b2ce31f5134913180c0261dd))
- provider registry, auto-launch wizard, assistant guardrail fixes ([5624729](https://github.com/NorthlandPositronics/Cogtrix/commit/56247295d7c37ba9d5fc4e22aede0797db50f729))
- publish multi-arch Docker image to GHCR ([d5cbcaa](https://github.com/NorthlandPositronics/Cogtrix/commit/d5cbcaa5763b79a63279f2f047823423daf34bc3))
- publish multi-arch Docker image to GHCR on main push ([a93dce2](https://github.com/NorthlandPositronics/Cogtrix/commit/a93dce2d438e43f707f3d48a7b5b26e5c8f612f8))
- randomize spinner phrase order on each start ([5c479dc](https://github.com/NorthlandPositronics/Cogtrix/commit/5c479dc12eec625ecb72a0428f5a5c7293b931fd))
- replace custom version-bump with release-please ([fecd792](https://github.com/NorthlandPositronics/Cogtrix/commit/fecd792c06565fd95c3d9b790d5b685f69d94c44))
- replace Panel with Rule+Padding for LLM response display ([c051c80](https://github.com/NorthlandPositronics/Cogtrix/commit/c051c80230195a1c308167ad76dc221f06ae6a97))
- restyle setup wizard with ANSI color and box-drawing UI ([541a06b](https://github.com/NorthlandPositronics/Cogtrix/commit/541a06be989da71294c12ad32436975133ef9826))
- restyle tool confirmation prompt labels and hotkeys ([9d0ca5c](https://github.com/NorthlandPositronics/Cogtrix/commit/9d0ca5c566d0d9ea9bbd6fea76f900f8f29c0165))
- sequential intro phrases then random fun phrases ([74500bf](https://github.com/NorthlandPositronics/Cogtrix/commit/74500bf50fa85f633a5b4c91f54a1c205067b265))
- show spinner during forced deep_think invocations ([236852a](https://github.com/NorthlandPositronics/Cogtrix/commit/236852a26fd66cf0037ae25ac6eeaa456c18cb75))
- start agent with only request_tools, all other tools on demand ([8099962](https://github.com/NorthlandPositronics/Cogtrix/commit/80999626b6d38b9ef8511b8d231cc8115688a8bf))
- tool-capable delegates and empty context validation ([ecfc45c](https://github.com/NorthlandPositronics/Cogtrix/commit/ecfc45c658261639326833067c49bcce98b630c0))

### Bug Fixes

- add execution phase so agent acts on analysis instead of just describing ([b16e3b9](https://github.com/NorthlandPositronics/Cogtrix/commit/b16e3b9116e2d314df16309d1bd28089d5a5375a))
- add nosec markers for Bandit false positives (B311, B110) ([a9062b8](https://github.com/NorthlandPositronics/Cogtrix/commit/a9062b87d62f6af4a0bef3cfce9d02a0ff4c81bc))
- add rollback to /mode switch to prevent state corruption on failure ([bea84ac](https://github.com/NorthlandPositronics/Cogtrix/commit/bea84ac1eaa30b30cac9d0e63a4f73063d6261db))
- add update_id tracking for Telegram deduplication and update docs ([e0223a4](https://github.com/NorthlandPositronics/Cogtrix/commit/e0223a4fb4ad57ec9c557426cdbd269c895f7ade))
- auto-activate on-demand tools and eliminate retry loops ([d691fd4](https://github.com/NorthlandPositronics/Cogtrix/commit/d691fd430b41ae25b0e8c7d3550d2bafec6cc925))
- break inner agent loop immediately after request_tools runs ([99de5ad](https://github.com/NorthlandPositronics/Cogtrix/commit/99de5ad4a51fd724be2e1c5e77d57b3126242797))
- category-aware Stage 2 framing for /think pipeline ([ca6d246](https://github.com/NorthlandPositronics/Cogtrix/commit/ca6d24656ce70309a3b26f2a054ffa6347561617))
- CI pipeline — dev dependencies, OSV-scanner, pyright error ([8954943](https://github.com/NorthlandPositronics/Cogtrix/commit/895494376ec73b74f1f0d54a515d578a97276619))
- CI pipeline — dev dependencies, OSV-scanner, pyright error ([65ef262](https://github.com/NorthlandPositronics/Cogtrix/commit/65ef2629ab989dda2ea315dd7797b97d01b29a5a))
- **ci:** move pytestmark below imports to fix ruff E402 ([82b8fec](https://github.com/NorthlandPositronics/Cogtrix/commit/82b8feca5193aac6b5db142facb631239d14341b))
- **ci:** resolve pyright type-ignore placement and bandit B310 warnings ([3a28a68](https://github.com/NorthlandPositronics/Cogtrix/commit/3a28a682eb69a29e8c08da3cf2df4201338ba967))
- **ci:** set bandit severity threshold to medium (-ll) ([1890edd](https://github.com/NorthlandPositronics/Cogtrix/commit/1890edd72768a8c38066370a3278c66b5fb4724f))
- clamp summarization index and harden JSON brace parsing ([659e4da](https://github.com/NorthlandPositronics/Cogtrix/commit/659e4daff0461f320b482556618fd242083a430a))
- classifier label normalization and tool output error filter ([89a951b](https://github.com/NorthlandPositronics/Cogtrix/commit/89a951b2e5c8ba1c9e23af0d0f8acf1893fca35b))
- clean up delegate tool alias resolution and remove dead max_depth ([69993de](https://github.com/NorthlandPositronics/Cogtrix/commit/69993de033e1fcf945f7728bf767eecebc75d964))
- correct /mode working memory display and add delegation tests ([70c6852](https://github.com/NorthlandPositronics/Cogtrix/commit/70c6852d362d2eadae4f21e109fb053dcae34e00))
- correct YAML output, RLock evict_idle, and guardrail false positives ([74c9fab](https://github.com/NorthlandPositronics/Cogtrix/commit/74c9fab1ca7af75443166a7c47dd196adffeac2e))
- detect unconfigured provider before LLM init ([bc3d5d8](https://github.com/NorthlandPositronics/Cogtrix/commit/bc3d5d83efd2f596778ec12c61fd273e4215c9b5))
- escape braces in deep_think prompts and guard escape flag to string context ([d999b7a](https://github.com/NorthlandPositronics/Cogtrix/commit/d999b7af0023e2736475ddbd7d6f74d06aebd1b1))
- give delegates all tools (active + on-demand) from the start ([acb316e](https://github.com/NorthlandPositronics/Cogtrix/commit/acb316ecfc9933205b7c422127a4f47695770754))
- graceful error messages for common configuration issues ([a4c9518](https://github.com/NorthlandPositronics/Cogtrix/commit/a4c9518d0da9597cdda65de989b1e92dfab94e5d))
- graceful error messages for common configuration issues ([62b14a4](https://github.com/NorthlandPositronics/Cogtrix/commit/62b14a417c2fb744750cba962b9cf2da5e6a6380))
- harden delegate tool review — docstring, tests, and exports ([2562d32](https://github.com/NorthlandPositronics/Cogtrix/commit/2562d327ba228d340cc167763b761762f614ab9c))
- harden input validation and add defensive checks (QA report) ([da3af41](https://github.com/NorthlandPositronics/Cogtrix/commit/da3af41deedc0fbec102d3aefdac6907cfeec2b8))
- harden input validation and correct misleading docstrings ([55c3d51](https://github.com/NorthlandPositronics/Cogtrix/commit/55c3d5139c02ab2f9bc968e70af1d0420b4963ba))
- harden memory system against multimodal content and ToolMessage corruption ([722ddda](https://github.com/NorthlandPositronics/Cogtrix/commit/722ddda26a5af00ea83d665ec1dd2b36721d1ddd))
- include search packages in Docker image ([93febfc](https://github.com/NorthlandPositronics/Cogtrix/commit/93febfc088bbd62f9aa60d5c8d3e56f92f5b5541))
- isolate config tests from env vars and use logging in config parser ([4c6df7d](https://github.com/NorthlandPositronics/Cogtrix/commit/4c6df7d2a87e7104abed51767b1a543c3bcea50c))
- keep request_tools meta-tool when all on-demand tools are activated ([e6e57b6](https://github.com/NorthlandPositronics/Cogtrix/commit/e6e57b6057ef5b491172e6abb279d548c73fcaff))
- lowercase Docker image name in release workflow ([2d30e82](https://github.com/NorthlandPositronics/Cogtrix/commit/2d30e820625d7a29700d0db25e631a876cc4527c))
- mode rollback, credit card regex, json wildcard, and docs ([ec35865](https://github.com/NorthlandPositronics/Cogtrix/commit/ec358656bb4b4bf722b579c2f4af46619472b6e8))
- optimize Dockerfile and harden .dockerignore ([049e8a3](https://github.com/NorthlandPositronics/Cogtrix/commit/049e8a39ff59497b9a6c8d68a0757553789a0784))
- pass force as boolean in version-bump ref update ([655059e](https://github.com/NorthlandPositronics/Cogtrix/commit/655059e4d986819b06ff89523979652ba4c9413a))
- pause spinner during deep_think progress output ([c58fc30](https://github.com/NorthlandPositronics/Cogtrix/commit/c58fc30cb92e7ff70f1d338ac438587ef9a8c0d3))
- persist full agent tool chain in history for iterative continuation ([52cfa75](https://github.com/NorthlandPositronics/Cogtrix/commit/52cfa75fffcc6b5bed977385e3958307fdf0ccfa))
- pin dependency versions and update requirements.txt ([4e34d90](https://github.com/NorthlandPositronics/Cogtrix/commit/4e34d90bbbed2096811a3418711c3975339207a8))
- preserve partial results in history for iterative refinement ([1e659e8](https://github.com/NorthlandPositronics/Cogtrix/commit/1e659e8fd2f30a65dea519cb3c4fadaeffd6e67b))
- prevent context window overflow during long agent runs ([2fe81c5](https://github.com/NorthlandPositronics/Cogtrix/commit/2fe81c5d568fb7471e5d879c7148dbf429cb2b94))
- prevent deep_think from producing meta-descriptions instead of answers ([72e7c88](https://github.com/NorthlandPositronics/Cogtrix/commit/72e7c88533e4902e1301d26452c9f9f126ccdda4))
- prevent delegation of tool-intensive tasks to LLM-only delegates ([e7678c1](https://github.com/NorthlandPositronics/Cogtrix/commit/e7678c1d15ef0df7d153aaf77a51de92b9e1df0a))
- prevent tool loss on release and mode switch ([ae15979](https://github.com/NorthlandPositronics/Cogtrix/commit/ae1597906dfe12c5e31733b01a63e58309b927e1))
- remove contradictory 'do not invent facts' from synthesis categories ([52c4650](https://github.com/NorthlandPositronics/Cogtrix/commit/52c4650efa4d7ea31ed38b9478ad49e2b84cf504))
- remove cycle-interval trigger from context compression ([5979284](https://github.com/NorthlandPositronics/Cogtrix/commit/5979284219ca4acc7ea2d2097e2c2453ef2d91f1))
- remove redundant guard and update query_json docstring ([4615f3c](https://github.com/NorthlandPositronics/Cogtrix/commit/4615f3c834f08c5443ede202de144c365be40d06))
- remove skip condition and add actions permission ([afdcf33](https://github.com/NorthlandPositronics/Cogtrix/commit/afdcf33b092fe5daf50c20c02d008aa69be0e1d7))
- rename release-please config to expected filename ([922ff3c](https://github.com/NorthlandPositronics/Cogtrix/commit/922ff3cc2abfa245701391f92c985d016ca6f1e7))
- resolve 12 bugs across orchestration, deep_think, and file_ops ([c5a78c5](https://github.com/NorthlandPositronics/Cogtrix/commit/c5a78c507c6032402d2c789df4e55e405c637f10))
- resolve delegation bugs with allowed_models and multimodal responses ([aed0051](https://github.com/NorthlandPositronics/Cogtrix/commit/aed00519236beeb6d16b02c180f618ded565c60d))
- resolve error filtering, display, and resource cleanup bugs ([5007dd1](https://github.com/NorthlandPositronics/Cogtrix/commit/5007dd1454a22ae020ecdad19952941b08f5850b))
- resolve MCP tool name closure, collision, unpacking, restart registry, and timeout bugs ([0f39a2d](https://github.com/NorthlandPositronics/Cogtrix/commit/0f39a2d41481f7f80f2b3886ea2ee0d85e01c71e))
- resolve multiple bugs across core agent and tool modules ([64958ed](https://github.com/NorthlandPositronics/Cogtrix/commit/64958ed84af1e393a1256773cb6171a535d7e172))
- resolve slash_cmds state desync, Rich markup injection, MCP restart, and token trim bugs ([50f65e4](https://github.com/NorthlandPositronics/Cogtrix/commit/50f65e42b80734c041b466be5f8aa6fbc7e89387))
- resolve stale agent_executor and false positive in execution phase ([d86e598](https://github.com/NorthlandPositronics/Cogtrix/commit/d86e5987826912332690a27bd015b2a8ade57d3f))
- resolve token estimation, message mutation, and timestamp bugs ([c092dc6](https://github.com/NorthlandPositronics/Cogtrix/commit/c092dc62aaf794f9d899c20266607ebd3b86deb3))
- resolve type-safety issues and apply formatting ([8e9e9cc](https://github.com/NorthlandPositronics/Cogtrix/commit/8e9e9ccc0e8e0567f623c787404b13f55c99f69c))
- show spinner during /think slash command ([ff2fdef](https://github.com/NorthlandPositronics/Cogtrix/commit/ff2fdefe354dc3e86ea8fe26bc39621f62098107))
- spinner line not clearing between frames ([f4a63e5](https://github.com/NorthlandPositronics/Cogtrix/commit/f4a63e5b57a922484412a5466601a64446058c33))
- stop injecting raw-JSON tool instructions into system prompt ([465a399](https://github.com/NorthlandPositronics/Cogtrix/commit/465a399581e77289788a12af33c01de5a2f5d71f))
- suppress primp native stderr leaking into spinner output ([55261b4](https://github.com/NorthlandPositronics/Cogtrix/commit/55261b414243575e52f58aeefa8df534f10abaf5))
- suppress pyright reportInvalidTypeForm on CogtrixState.messages ([d4b1909](https://github.com/NorthlandPositronics/Cogtrix/commit/d4b190943db8397652cfc393490cbb7fdd7a28b4))
- suppress ruff F401 for TYPE_CHECKING-only import in safety.py ([7c5af4c](https://github.com/NorthlandPositronics/Cogtrix/commit/7c5af4c51977bbd4dfe8e1835add8e428e376c9c))
- switch rollback, json multi-bracket paths, and doc corrections ([a00f5f3](https://github.com/NorthlandPositronics/Cogtrix/commit/a00f5f3d59250606cdeeea9f0c561880540f5c89))
- sync tools list after outer agent rebuild ([3876751](https://github.com/NorthlandPositronics/Cogtrix/commit/387675185a1e63a8ce709585cc852bd0968a0308))
- tool confirmation panel rendering and bare slash crash ([437c47b](https://github.com/NorthlandPositronics/Cogtrix/commit/437c47b4c61a3050f3c36493a8669215151a4194))
- use native ARM64 Linux runner for Docker build ([925d95e](https://github.com/NorthlandPositronics/Cogtrix/commit/925d95e7bd128886002cb8ed10d6d752e45840ac))
- use native runners for Docker builds, remove QEMU emulation ([b611365](https://github.com/NorthlandPositronics/Cogtrix/commit/b611365dca6fa67bb634643ea55fa134bb4abf2d))
- use temp files for GitHub API blob creation in version-bump ([082dfa0](https://github.com/NorthlandPositronics/Cogtrix/commit/082dfa089bd9e39230947374bdebe625a3cce2e0))
- validate relative paths against cwd to catch symlink traversal ([b459abc](https://github.com/NorthlandPositronics/Cogtrix/commit/b459abc04cdd75913261cf39875b38a70a52d64b))
- wire LLM into new memory managers on mode/session switch ([745dd9b](https://github.com/NorthlandPositronics/Cogtrix/commit/745dd9bd6b068c806b7cb4039b4e4676cc8defb7))

### Performance Improvements

- parallelize context compression LLM calls ([d565a47](https://github.com/NorthlandPositronics/Cogtrix/commit/d565a4784f99194db215d0a19c5d8fdfef350b22))

### Documentation

- add detailed Telegram assistant guide ([ed4fad0](https://github.com/NorthlandPositronics/Cogtrix/commit/ed4fad0a11c57d6a1b8a2d19173f6ae4ffbb5066))
- add YAML examples, document allowed_models, and add /delegate command ([07895e4](https://github.com/NorthlandPositronics/Cogtrix/commit/07895e4328f5fb6f943213b696ed8f59c6a4bf82))
- document assistant mode in README, ARCHITECTURE, and DEVELOPMENT ([a6ab6b5](https://github.com/NorthlandPositronics/Cogtrix/commit/a6ab6b5d3ff78c9a4c908b2f2d162f8df25e1601))
- document hybrid memory system across all guides ([0353a6a](https://github.com/NorthlandPositronics/Cogtrix/commit/0353a6a3808e882ba44bf5a265983e462d1e8119))
- document tool presets and on-demand auto-activation ([50b5372](https://github.com/NorthlandPositronics/Cogtrix/commit/50b537235a162e15be218f1f531c3f7dce519111))
- fix clone URL, tool counts, and missing telegram references ([4b0c6ec](https://github.com/NorthlandPositronics/Cogtrix/commit/4b0c6ec4d3c5e81e8d75c9e74099472f2ef3409a))
- fix stray backtick, outdated description, and wrong linter commands ([47be75a](https://github.com/NorthlandPositronics/Cogtrix/commit/47be75a3916ac964eea4419dfdad357898608cf4))
- improve documentation for newcomers and consistency ([99d2916](https://github.com/NorthlandPositronics/Cogtrix/commit/99d29161b757d83b2b509c829a84058e56c084b2))
- improve getting-started guide and document search provider setup ([2c15462](https://github.com/NorthlandPositronics/Cogtrix/commit/2c15462374d14174aa5cdb73d5d009d359cec510))
- improve getting-started guide and document search provider setup ([7d1c036](https://github.com/NorthlandPositronics/Cogtrix/commit/7d1c036f74212f6bde9cdb34038b55b5f8789ea0)), closes [#22](https://github.com/NorthlandPositronics/Cogtrix/issues/22)
- overhaul documentation for clarity and consistency ([10ad02c](https://github.com/NorthlandPositronics/Cogtrix/commit/10ad02cda8608a513eea7ae7e3d0ed1bad1917cf))
- polish README and fix misleading RAG directory example ([ae93500](https://github.com/NorthlandPositronics/Cogtrix/commit/ae93500649a79cb3f0ff5325fd099e19e9916425))
- standardize all config examples to YAML across guides ([d033bd7](https://github.com/NorthlandPositronics/Cogtrix/commit/d033bd7e28487fbe0f66fa8a217fbfe6b12d4638))
- update context compression docs for parallelization and model override ([b4132e2](https://github.com/NorthlandPositronics/Cogtrix/commit/b4132e2be6b8a3349d7dd044b80b66e083f5ffc8))
- update DEEPTHINK.md quotes to match tightened system prompt ([c02d3a0](https://github.com/NorthlandPositronics/Cogtrix/commit/c02d3a0a224f616f6a9b42f50c987ac5aff5682e))
- update documentation to reflect on-demand tool loading ([ea36469](https://github.com/NorthlandPositronics/Cogtrix/commit/ea3646931f394f11cf6b508ea501d01c9d8cab5f))
- update tool_instructions description in CONFIGURATION.md ([835af41](https://github.com/NorthlandPositronics/Cogtrix/commit/835af41d858471503b3fc723af315e9e69e68d02))

## [0.1.2](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.1...v0.1.2) (2026-02-16)

### Bug Fixes

- include search packages in Docker image ([6b6eb02](https://github.com/NorthlandPositronics/Cogtrix/commit/6b6eb02aa0bbbe74adf4e5bbb940ece4859e6501))

### Documentation

- improve getting-started guide and document search provider setup ([bbf113d](https://github.com/NorthlandPositronics/Cogtrix/commit/bbf113d00238074218b8be83176045c28a3ba3a7))
- improve getting-started guide and document search provider setup ([16f6a92](https://github.com/NorthlandPositronics/Cogtrix/commit/16f6a9256aadd889b1ee92e25f9ec14c9ab47c42)), closes [#22](https://github.com/NorthlandPositronics/Cogtrix/issues/22)

## [0.1.1](https://github.com/NorthlandPositronics/Cogtrix/compare/v0.1.0...v0.1.1) (2026-02-16)

### Features

- activity indicator (spinner) during LLM processing ([0d9b4ac](https://github.com/NorthlandPositronics/Cogtrix/commit/0d9b4ac3a2ecf944bf65bf2bf40fde1a221980a9))
- auto version bump and release on merge to main ([f420524](https://github.com/NorthlandPositronics/Cogtrix/commit/f420524b6ada84a26de3abb1ab3d854c5ad21c0e))
- double spinner phrases, gradient color, trailing space ([7cc2067](https://github.com/NorthlandPositronics/Cogtrix/commit/7cc2067f8317e0743bb2227d47d40721bf806dcd))
- expand spinner to 80 phrases with humorous tech messages ([fd9d569](https://github.com/NorthlandPositronics/Cogtrix/commit/fd9d5699b84c9a1f13c481dd4d22c64450c0c3cb))
- hybrid /think pipeline with category-aware prompts ([6308a86](https://github.com/NorthlandPositronics/Cogtrix/commit/6308a86f28e58c8d225a79fb7b461c17b5b40fe5))
- publish multi-arch Docker image to GHCR ([d4325e4](https://github.com/NorthlandPositronics/Cogtrix/commit/d4325e41254a128e4588fbc173ce6cdee638e611))
- publish multi-arch Docker image to GHCR on main push ([1a28ea6](https://github.com/NorthlandPositronics/Cogtrix/commit/1a28ea60166a037305c5ba19bb1478bbaac3de0a))
- randomize spinner phrase order on each start ([0e0b107](https://github.com/NorthlandPositronics/Cogtrix/commit/0e0b10778577dc59a367c4a21c8fe522ccb5de69))
- replace custom version-bump with release-please ([5790406](https://github.com/NorthlandPositronics/Cogtrix/commit/57904061c3816c6770de27847c37585616b1ea0e))
- sequential intro phrases then random fun phrases ([1f5bbf9](https://github.com/NorthlandPositronics/Cogtrix/commit/1f5bbf9d207be12872cf719216b245a03cfe9b88))
- show spinner during forced deep_think invocations ([6d6679e](https://github.com/NorthlandPositronics/Cogtrix/commit/6d6679ebd0bd503839a81c239dab8a94b6f2803f))

### Bug Fixes

- add nosec markers for Bandit false positives (B311, B110) ([739449b](https://github.com/NorthlandPositronics/Cogtrix/commit/739449b6b63af7feb87dc708790792716b6805aa))
- category-aware Stage 2 framing for /think pipeline ([875f085](https://github.com/NorthlandPositronics/Cogtrix/commit/875f08541712645131531f38c699716e8d83e7ca))
- CI pipeline — dev dependencies, OSV-scanner, pyright error ([3e21dd1](https://github.com/NorthlandPositronics/Cogtrix/commit/3e21dd16079c971d9dcb97334cefa32db126d8b1))
- CI pipeline — dev dependencies, OSV-scanner, pyright error ([93ac20a](https://github.com/NorthlandPositronics/Cogtrix/commit/93ac20aa415dcfcfd22c1690370a6085d8c78b7a))
- classifier label normalization and tool output error filter ([f87821f](https://github.com/NorthlandPositronics/Cogtrix/commit/f87821fe0fc8e2779684506f3caafc7317d2f4dd))
- detect unconfigured provider before LLM init ([8425b93](https://github.com/NorthlandPositronics/Cogtrix/commit/8425b9331a72ba46262c7ba560fbdf25038a3027))
- graceful error messages for common configuration issues ([a9922e3](https://github.com/NorthlandPositronics/Cogtrix/commit/a9922e30e08042bcf967686207aaf9943e048e11))
- graceful error messages for common configuration issues ([00e8128](https://github.com/NorthlandPositronics/Cogtrix/commit/00e81281ac993d366a2231f8f81f243bad0c6e5c))
- pass force as boolean in version-bump ref update ([d11d4dd](https://github.com/NorthlandPositronics/Cogtrix/commit/d11d4dd70051f9a2589a91aa61bda0bc69a896e4))
- pause spinner during deep_think progress output ([a8dd126](https://github.com/NorthlandPositronics/Cogtrix/commit/a8dd126d29843065ac127e067b2306d0d6f498be))
- prevent deep_think from producing meta-descriptions instead of answers ([a3fd699](https://github.com/NorthlandPositronics/Cogtrix/commit/a3fd699104d9584163b4f6bcf8eb0349dc9c85be))
- remove contradictory 'do not invent facts' from synthesis categories ([1f2e65e](https://github.com/NorthlandPositronics/Cogtrix/commit/1f2e65eecfafed94cabb24a1ea0b42cc86ba2de6))
- remove skip condition and add actions permission ([4caffd0](https://github.com/NorthlandPositronics/Cogtrix/commit/4caffd02e10ab3c4a6eefd68ed1cad89d05ec6a2))
- rename release-please config to expected filename ([0bb0a0c](https://github.com/NorthlandPositronics/Cogtrix/commit/0bb0a0cc61ea8f964841647153a5120d707cae86))
- show spinner during /think slash command ([bbb5b56](https://github.com/NorthlandPositronics/Cogtrix/commit/bbb5b564f55fb65864fa7cd3c83805c1eb8a10f1))
- spinner line not clearing between frames ([2236607](https://github.com/NorthlandPositronics/Cogtrix/commit/2236607f35450dd05494e941b392d905f38675ef))
- suppress primp native stderr leaking into spinner output ([2fa5531](https://github.com/NorthlandPositronics/Cogtrix/commit/2fa55310181a2261d373323c0417ac1c266f5a2d))
- suppress ruff F401 for TYPE_CHECKING-only import in safety.py ([d9ec409](https://github.com/NorthlandPositronics/Cogtrix/commit/d9ec409294dbac181673129e7ca9c448e1d1a0df))
- use native ARM64 Linux runner for Docker build ([d95b140](https://github.com/NorthlandPositronics/Cogtrix/commit/d95b1408da9f96bfedcc741fc49c1827540b1393))
- use temp files for GitHub API blob creation in version-bump ([a2bc546](https://github.com/NorthlandPositronics/Cogtrix/commit/a2bc5464e9fd3afff44352dbcf05018ff0b8e6d8))

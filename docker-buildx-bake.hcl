group "default" {
  targets = ["cogtrix"]
}

target "cogtrix" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64", "linux/arm64"]
  tags       = ["ghcr.io/northlandpositronics/cogtrix:latest"]
  attest = [
    "type=provenance,mode=max",
    "type=sbom",
  ]
}

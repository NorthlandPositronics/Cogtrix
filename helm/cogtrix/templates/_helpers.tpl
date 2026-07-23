{{/*
Expand the name of the chart.
*/}}
{{- define "cogtrix.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "cogtrix.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "cogtrix.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "cogtrix.labels" -}}
helm.sh/chart: {{ include "cogtrix.chart" . }}
{{ include "cogtrix.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "cogtrix.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cogtrix.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "cogtrix.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "cogtrix.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "cogtrix.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}
{{- end }}

{{/*
Return the proper WAHA sidecar image name
*/}}
{{- define "cogtrix.wahaImage" -}}
{{- if .Values.waha.image.digest -}}
{{- printf "%s@%s" .Values.waha.image.repository .Values.waha.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.waha.image.repository .Values.waha.image.tag -}}
{{- end -}}
{{- end }}

{{/*
Return the proper init-container image name (digest-pinned when provided)
*/}}
{{- define "cogtrix.initImage" -}}
{{- if .Values.initContainer.image.digest -}}
{{- printf "%s@%s" .Values.initContainer.image.repository .Values.initContainer.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.initContainer.image.repository .Values.initContainer.image.tag -}}
{{- end -}}
{{- end }}

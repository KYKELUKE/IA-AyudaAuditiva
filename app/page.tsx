"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Brain, Upload, BarChart3, User, Heart, Mic, Pause } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

// Datos de ejemplo para el historial emocional
const emotionalHistory = [
  { date: "2024-01-15", emotion: "Alegría", score: 8 },
  { date: "2024-01-16", emotion: "Neutral", score: 6 },
  { date: "2024-01-17", emotion: "Tristeza", score: 3 },
  { date: "2024-01-18", emotion: "Ansiedad", score: 4 },
  { date: "2024-01-19", emotion: "Alegría", score: 7 },
]

const emotionMessages = {
  Alegría: "¡Qué maravilloso! Tu energía positiva es contagiosa. Sigue cultivando esos momentos de felicidad.",
  Tristeza:
    "Es normal sentir tristeza a veces. Recuerda que estos sentimientos son temporales y está bien pedir ayuda.",
  Ansiedad: "Respira profundo. La ansiedad puede ser abrumadora, pero hay técnicas que pueden ayudarte a manejarla.",
  Neutral: "Un estado equilibrado es valioso. Aprovecha esta calma para reflexionar y cuidar tu bienestar.",
  Enojo: "El enojo es una emoción válida. Trata de identificar qué lo causa y busca formas saludables de expresarlo.",
}

export default function EmotionalSupportAssistant() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<{
    emotion: string
    confidence: number
    message: string
  } | null>(null)
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [isRecording, setIsRecording] = useState(false)

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && (file.type === "audio/wav" || file.type === "audio/mp3" || file.type === "audio/mpeg")) {
      setSelectedFile(file)
    } else {
      alert("Por favor, selecciona un archivo de audio válido (.wav o .mp3)")
    }
  }

  const analyzeEmotion = async () => {
    if (!selectedFile) {
      alert("Por favor, selecciona un archivo de audio primero")
      return
    }

    setIsAnalyzing(true)

    // Simulación de análisis (aquí conectarías con tu API de Flask)
    setTimeout(() => {
      const emotions = ["Alegría", "Tristeza", "Ansiedad", "Neutral", "Enojo"]
      const randomEmotion = emotions[Math.floor(Math.random() * emotions.length)]
      const confidence = Math.floor(Math.random() * 30) + 70 // 70-100%

      setAnalysisResult({
        emotion: randomEmotion,
        confidence,
        message: emotionMessages[randomEmotion as keyof typeof emotionMessages],
      })
      setIsAnalyzing(false)
    }, 3000)
  }

  const toggleRecording = () => {
    setIsRecording(!isRecording)
    // Aquí implementarías la lógica de grabación
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-green-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-purple-100 sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-r from-purple-100 to-blue-100 rounded-full">
              <Brain className="h-8 w-8 text-purple-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Asistente de Apoyo Emocional</h1>
              <p className="text-sm text-gray-600">Análisis emocional con IA</p>
            </div>
          </div>
          <Button
            variant={isLoggedIn ? "outline" : "default"}
            onClick={() => setIsLoggedIn(!isLoggedIn)}
            className="bg-purple-600 hover:bg-purple-700 text-white"
          >
            <User className="h-4 w-4 mr-2" />
            {isLoggedIn ? "Mi Perfil" : "Iniciar Sesión"}
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Upload Section */}
        <Card className="mb-8 border-purple-200 shadow-lg">
          <CardHeader className="text-center">
            <CardTitle className="flex items-center justify-center gap-2 text-2xl text-gray-800">
              <Heart className="h-6 w-6 text-pink-500" />
              Análisis de Emociones por Voz
            </CardTitle>
            <CardDescription className="text-gray-600">
              Sube un archivo de audio o graba tu voz para analizar tu estado emocional
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* File Upload */}
            <div className="space-y-2">
              <Label htmlFor="audio-file" className="text-gray-700 font-medium">
                Subir archivo de audio (.wav, .mp3)
              </Label>
              <div className="flex items-center gap-4">
                <Input
                  id="audio-file"
                  type="file"
                  accept=".wav,.mp3,audio/*"
                  onChange={handleFileUpload}
                  className="border-purple-200 focus:border-purple-400"
                />
                <Button
                  variant="outline"
                  onClick={toggleRecording}
                  className={`${isRecording ? "bg-red-100 border-red-300" : "border-purple-200"}`}
                >
                  {isRecording ? <Pause className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                  {isRecording ? "Detener" : "Grabar"}
                </Button>
              </div>
            </div>

            {selectedFile && (
              <Alert className="border-green-200 bg-green-50">
                <Upload className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-700">
                  Archivo seleccionado: {selectedFile.name}
                </AlertDescription>
              </Alert>
            )}

            {/* Analyze Button */}
            <Button
              onClick={analyzeEmotion}
              disabled={!selectedFile || isAnalyzing}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-6 text-lg font-semibold"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Analizando emoción...
                </>
              ) : (
                <>
                  <Brain className="h-5 w-5 mr-2" />
                  Analizar Emoción
                </>
              )}
            </Button>

            {isAnalyzing && (
              <div className="space-y-2">
                <Progress value={66} className="w-full" />
                <p className="text-sm text-gray-600 text-center">
                  Procesando audio y analizando patrones emocionales...
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        {analysisResult && (
          <Card className="mb-8 border-green-200 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xl text-gray-800">
                <BarChart3 className="h-5 w-5 text-green-600" />
                Resultado del Análisis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg">
                <h3 className="text-3xl font-bold text-gray-800 mb-2">{analysisResult.emotion}</h3>
                <p className="text-lg text-gray-600 mb-4">Confianza: {analysisResult.confidence}%</p>
                <div className="bg-white p-4 rounded-lg border border-green-200">
                  <p className="text-gray-700 leading-relaxed">{analysisResult.message}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* History Section */}
        {isLoggedIn && (
          <Card className="mb-8 border-blue-200 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2 text-xl text-gray-800">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                  Historial Emocional
                </span>
                <Button variant="outline" onClick={() => setShowHistory(!showHistory)} className="border-blue-200">
                  {showHistory ? "Ocultar" : "Mostrar"} Gráfico
                </Button>
              </CardTitle>
            </CardHeader>
            {showHistory && (
              <CardContent>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={emotionalHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                      <XAxis dataKey="date" stroke="#6b7280" />
                      <YAxis domain={[0, 10]} stroke="#6b7280" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#f8fafc",
                          border: "1px solid #e2e8f0",
                          borderRadius: "8px",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#8b5cf6"
                        strokeWidth={3}
                        dot={{ fill: "#8b5cf6", strokeWidth: 2, r: 6 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            )}
          </Card>
        )}

        {/* Disclaimer */}
        <Alert className="border-amber-200 bg-amber-50">
          <Heart className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-amber-800">
            <strong>Importante:</strong> Este sistema es solo un apoyo emocional y no sustituye el diagnóstico
            profesional. Consulta siempre a un especialista en salud mental para obtener ayuda profesional.
          </AlertDescription>
        </Alert>
      </main>

      {/* Footer */}
      <footer className="bg-white/80 backdrop-blur-sm border-t border-purple-100 mt-12">
        <div className="container mx-auto px-4 py-6 text-center">
          <p className="text-gray-600">© 2024 Asistente de Apoyo Emocional - Cuidando tu bienestar mental</p>
        </div>
      </footer>
    </div>
  )
}

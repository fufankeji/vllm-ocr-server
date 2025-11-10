import { useState } from 'react';
import { FileUploadPanel } from './components/FileUploadPanel';
import { ResultsPanel } from './components/ResultsPanel';
import { Toaster } from './components/ui/sonner';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<any>(null);
  const [fullMarkdown, setFullMarkdown] = useState<string>('');

  const handleStartAnalysis = async () => {
    if (!selectedFile || !selectedModel) {
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);
    setResults(null);
    setFullMarkdown('');

    try {
      // 调用真实的后端API
      const formData = new FormData();
      formData.append('file', selectedFile);

      // Map frontend model names to backend model names
      const modelMap: { [key: string]: string } = {
        'mineru': 'mineru',
        'deepseek': 'deepseek',
        'deepseek-ocr': 'deepseek',
        'paddleocr': 'paddleocr',
        'paddleocr-vl': 'paddleocr'
      };

      // Debug logging for model selection
      console.log('🔍 Frontend model debugging:');
      console.log('  Selected model (raw):', selectedModel);
      console.log('  Selected model (lowercase):', selectedModel.toLowerCase());
      const backendModel = modelMap[selectedModel.toLowerCase()] || 'mineru';
      console.log('  Mapped backend model:', backendModel);

      formData.append('model', backendModel);
      console.log('  FormData model field added:', backendModel);

      formData.append('options', JSON.stringify({
        backend: 'vlm-vllm-async-engine',
        enable_ocr: true,
        language: 'ch',
        device: 'cuda:3'
      }));

      // 模拟进度更新
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 10, 90));
      }, 300);

      // 调用后端API
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/ocr/analyze`, {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        // 使用后端返回的真实结果
        console.log('API Response:', data);
        console.log('Images count:', data.results?.images?.length || 0);
        console.log('Tables count:', data.results?.tables?.length || 0);
        console.log('Formulas count:', data.results?.formulas?.length || 0);
        setResults(data.results);
        setFullMarkdown(data.fullMarkdown || '');
      } else {
        throw new Error(data.error || 'OCR分析失败');
      }

      setIsAnalyzing(false);
    } catch (error) {
      console.error('OCR分析错误:', error);
      setIsAnalyzing(false);
      setProgress(0);

      // 显示错误提示
      const errorMessage = error instanceof Error ? error.message : '未知错误';
      alert(`OCR分析失败: ${errorMessage}\n\n请确保后端服务正在运行 (http://localhost:8000)`);

      // 如果API调用失败，回退到模拟数据
      setResults({
        text: generateMockTextResults(selectedModel),
        tables: generateMockTableResults(),
        formulas: generateMockFormulaResults(),
        images: generateMockImageResults(),
        handwritten: generateMockHandwrittenResults(),
        performance: generateMockPerformanceData(selectedModel),
      });
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden relative">
      <Toaster />
      
      {/* Animated background effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>
      
      {/* Header */}
      <header className="glass flex-shrink-0 border-b border-white/10 relative z-10">
        <div className="max-w-[1800px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 animate-gradient">
                赋范空间公开课 by 木羽Cheney
              </h1>
              <p className="text-cyan-400/80 mt-0.5">OCR 智能分析平台 · 支持多种模型的文档识别与分析</p>
            </div>
            <button className="px-6 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white hover:from-cyan-400 hover:to-blue-400 transition-all glow-cyan">
              点击获取课程优惠
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden relative z-10">
        <div className="h-full max-w-[1800px] mx-auto px-6 py-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
            {/* Left Panel */}
            <FileUploadPanel
              selectedFile={selectedFile}
              setSelectedFile={setSelectedFile}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              isAnalyzing={isAnalyzing}
              progress={progress}
              onStartAnalysis={handleStartAnalysis}
            />

            {/* Right Panel */}
            <ResultsPanel
              results={results}
              isAnalyzing={isAnalyzing}
              selectedFile={selectedFile}
              fullMarkdown={fullMarkdown}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

// Mock data generators
function generateMockTextResults(model: string) {
  const texts = {
    'MinerU': '这是一份关于人工智能发展的重要文档。文档中详细介绍了深度学习技术在图像识别领域的应用。通过卷积神经网络(CNN)和循环神经网络(RNN)的结合，我们可以实现更加精确的文本识别。实验结果表明，该方法在多个基准数据集上都取得了优异的性能。',
    'PaddleOCR-VL': '本文档探讨了光学字符识别技术的最新进展。OCR 技术已经广泛应用于文档数字化、自动化办公等领域。随着深度学习的发展，OCR 的准确率有了显著提升。特别是在处理复杂背景和多语言混合文本时，新一代 OCR 模型展现出了强大的能力。',
    'DeepSeek-OCR': '光学字符识别（OCR）是计算机视觉的重要分支。现代 OCR 系统能够处理印刷体、手写体以及复杂排版的文档。通过预训练模型和迁移学习，OCR 技术在识别准确率和处理速度上都有了质的飞跃。未来，OCR 将在智能办公、自动驾驶等领域发挥更大作用。'
  };
  
  return {
    fullText: texts[model as keyof typeof texts] || texts['MinerU'],
    keywords: ['人工智能', '深度学习', 'OCR', '图像识别', '神经网络'],
    confidence: Math.random() * 10 + 90
  };
}

function generateMockTableResults() {
  return [
    {
      title: '表格 1: 模型性能对比',
      headers: ['模型名称', '准确率 (%)', '处理时间 (秒)', '内存占用 (MB)'],
      rows: [
        ['MinerU', '96.5', '2.3', '512'],
        ['PaddleOCR-VL', '95.8', '1.8', '384'],
        ['DeepSeek-OCR', '97.2', '2.1', '448']
      ]
    }
  ];
}

function generateMockFormulaResults() {
  return [
    { formula: 'E = mc²', description: '质能方程' },
    { formula: '∫₀^∞ e^(-x²) dx = √π/2', description: '高斯积分' },
    { formula: 'f(x) = ax² + bx + c', description: '二次函数' }
  ];
}

function generateMockImageResults() {
  return [
    { type: '图表', description: '柱状图：各模型性能对比', confidence: 94.5 },
    { type: '示意图', description: '神经网络结构图', confidence: 92.3 }
  ];
}

function generateMockHandwrittenResults() {
  return {
    detected: true,
    text: '手写笔记：重要会议记录',
    confidence: 88.7
  };
}

function generateMockPerformanceData(model: string) {
  const baseData = {
    'MinerU': { accuracy: 96.5, speed: 2.3, memory: 512 },
    'PaddleOCR-VL': { accuracy: 95.8, speed: 1.8, memory: 384 },
    'DeepSeek-OCR': { accuracy: 97.2, speed: 2.1, memory: 448 }
  };

  return baseData[model as keyof typeof baseData] || baseData['MinerU'];
}

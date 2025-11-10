import { Upload, FileText, AlertCircle } from 'lucide-react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { ScrollArea } from './ui/scroll-area';
import { toast } from 'sonner@2.0.3';
import { useState, useRef } from 'react';

interface FileUploadPanelProps {
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  isAnalyzing: boolean;
  progress: number;
  onStartAnalysis: () => void;
}

export function FileUploadPanel({
  selectedFile,
  setSelectedFile,
  selectedModel,
  setSelectedModel,
  isAnalyzing,
  progress,
  onStartAnalysis,
}: FileUploadPanelProps) {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (file: File) => {
    const validTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
    
    if (!validTypes.includes(file.type)) {
      toast.error('不支持的文件格式', {
        description: '请上传 PDF 或图片文件（PNG、JPG、JPEG、WEBP）'
      });
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('文件过大', {
        description: '文件大小不能超过 10MB'
      });
      return;
    }

    setSelectedFile(file);
    toast.success('文件上传成功', {
      description: `已选择: ${file.name}`
    });
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleStartClick = () => {
    if (!selectedFile) {
      toast.error('请先上传文件');
      return;
    }
    if (!selectedModel) {
      toast.error('请选择 OCR 模型');
      return;
    }
    onStartAnalysis();
  };

  return (
    <ScrollArea className="h-full">
      <div className="space-y-4 pr-4">
        {/* File Upload Card */}
        <Card className="p-4 glass border-cyan-500/20 hover:border-cyan-500/40 transition-all">
          <h2 className="text-cyan-400 mb-3 flex items-center gap-2">
            <Upload className="w-5 h-5" />
            文件上传
          </h2>
          
          <div
            className={`relative border-2 border-dashed rounded-lg p-6 transition-all ${
              dragActive
                ? 'border-cyan-500 bg-cyan-500/10 glow-cyan'
                : 'border-cyan-500/30 bg-slate-900/50 hover:border-cyan-500/50 hover:bg-slate-900/70'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf,.png,.jpg,.jpeg,.webp"
              onChange={handleFileInputChange}
              disabled={isAnalyzing}
            />

            <div className="flex flex-col items-center justify-center text-center">
              <Upload className="w-10 h-10 text-cyan-400 mb-3" />
              <p className="text-slate-200 mb-1">
                请上传 PDF 或图片文件进行 OCR 分析
              </p>
              <p className="text-slate-400 text-sm mb-3">
                支持格式: PDF, PNG, JPG, JPEG, WEBP (最大 500MB)
              </p>
              <Button
                onClick={() => fileInputRef.current?.click()}
                disabled={isAnalyzing}
                variant="outline"
                size="sm"
                className="border-cyan-400 text-cyan-300 font-medium hover:bg-cyan-500/10 hover:border-cyan-400 hover:text-cyan-200"
              >
                <Upload className="w-4 h-4 mr-2" />
                选择文件
              </Button>
            </div>
          </div>

          {selectedFile && (
            <div className="mt-3 p-3 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-lg glow-cyan">
              <div className="flex items-start gap-3">
                <FileText className="w-5 h-5 text-cyan-400 mt-0.5" />
                <div className="flex-1">
                  <p className="text-slate-200">{selectedFile.name}</p>
                  <div className="flex gap-4 mt-1 text-sm text-slate-400">
                    <span>大小: {(selectedFile.size / 1024).toFixed(2)} KB</span>
                    <span>类型: {selectedFile.type.split('/')[1].toUpperCase()}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Model Selection Card */}
        <Card className="p-4 glass border-blue-500/20 hover:border-blue-500/40 transition-all">
          <h2 className="text-blue-400 mb-3">模型选择</h2>
          
          <div className="space-y-3">
            <div>
              <label className="block text-slate-300 mb-2">
                选择 OCR 处理模型
              </label>
              <Select
                value={selectedModel}
                onValueChange={setSelectedModel}
                disabled={isAnalyzing}
              >
                <SelectTrigger className="bg-slate-900/50 border-blue-500/30 text-cyan-300 font-medium hover:border-blue-500/50 data-[placeholder]:text-cyan-300">
                  <SelectValue placeholder="请选择 OCR 模型" />
                </SelectTrigger>
                <SelectContent className="bg-slate-900 border-blue-500/30">
                  <SelectItem value="mineru" className="text-cyan-200 focus:bg-blue-500/20 focus:text-cyan-300 cursor-pointer font-medium">MinerU - 高精度文档识别</SelectItem>
                  <SelectItem value="paddleocr" className="text-cyan-200 focus:bg-blue-500/20 focus:text-cyan-300 cursor-pointer font-medium">PaddleOCR-VL - 快速处理</SelectItem>
                  <SelectItem value="deepseek" className="text-cyan-200 focus:bg-blue-500/20 focus:text-cyan-300 cursor-pointer font-medium">DeepSeek-OCR - 综合性能</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Alert className="bg-blue-500/10 border-blue-500/30 text-blue-300">
              <AlertCircle className="h-4 w-4 text-blue-400" />
              <AlertDescription className="text-blue-200">
                不同模型在准确率、速度和资源占用上各有特点，请根据需求选择合适的模型。
              </AlertDescription>
            </Alert>
          </div>
        </Card>

        {/* Start Analysis Card */}
        <Card className="p-4 glass border-purple-500/20 hover:border-purple-500/40 transition-all">
          <Button
            className="w-full bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500 hover:from-cyan-400 hover:via-blue-400 hover:to-purple-400 text-white border-0 glow-cyan"
            onClick={handleStartClick}
            disabled={isAnalyzing || !selectedFile || !selectedModel}
          >
            {isAnalyzing ? '分析中...' : '开始 OCR 分析'}
          </Button>

          {isAnalyzing && (
            <div className="mt-3 space-y-2">
              <div className="flex justify-between text-sm text-cyan-400">
                <span>分析进度</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} className="bg-slate-800 [&>div]:bg-gradient-to-r [&>div]:from-cyan-500 [&>div]:to-purple-500" />
            </div>
          )}
        </Card>
      </div>
    </ScrollArea>
  );
}

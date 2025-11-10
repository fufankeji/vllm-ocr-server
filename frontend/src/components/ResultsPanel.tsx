import { useState } from 'react';
import { Card } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Download, FileText, Table2, Calculator, Image, PenTool, BarChart3, Maximize2 } from 'lucide-react';
import { toast } from 'sonner@2.0.3';
import { Alert, AlertDescription } from './ui/alert';
import ReactMarkdown from 'react-markdown';

interface ResultsPanelProps {
  results: any;
  isAnalyzing: boolean;
  selectedFile: File | null;
  fullMarkdown?: string;
}

export function ResultsPanel({ results, isAnalyzing, selectedFile, fullMarkdown }: ResultsPanelProps) {
  const [selectedImage, setSelectedImage] = useState<{src: string, alt: string} | null>(null);

  const handleExport = (format: 'csv' | 'pdf') => {
    if (!results) return;
    
    toast.success(`导出成功`, {
      description: `OCR 结果已导出为 ${format.toUpperCase()} 格式`
    });
  };

  if (!selectedFile && !results && !isAnalyzing) {
    return (
      <Card className="h-full flex items-center justify-center glass border-cyan-500/20">
        <div className="text-center">
          <FileText className="w-16 h-16 text-cyan-400 mx-auto mb-4 opacity-50" />
          <h3 className="text-cyan-400 mb-2">等待分析</h3>
          <p className="text-slate-400">
            请上传文件并选择模型开始 OCR 分析
          </p>
        </div>
      </Card>
    );
  }

  if (isAnalyzing) {
    return (
      <Card className="h-full flex items-center justify-center glass border-blue-500/20">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4 glow-cyan"></div>
          <h3 className="text-cyan-400 mb-2">正在分析中</h3>
          <p className="text-slate-400">
            OCR 模型正在处理您的文档，请稍候...
          </p>
        </div>
      </Card>
    );
  }

  if (!results) {
    return null;
  }

  return (
    <div className="h-full flex flex-col gap-3">
      {/* Export Options */}
      <Card className="p-3 flex-shrink-0 glass border-cyan-500/20">
        <div className="flex items-center justify-between">
          <h2 className="text-cyan-400 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            分析结果
          </h2>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleExport('csv')}
              className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 hover:border-cyan-500"
            >
              <Download className="w-4 h-4 mr-2" />
              导出 CSV
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => handleExport('pdf')}
              className="border-blue-500/50 text-blue-400 hover:bg-blue-500/10 hover:border-blue-500"
            >
              <Download className="w-4 h-4 mr-2" />
              导出 PDF
            </Button>
          </div>
        </div>
      </Card>

      {/* Results Tabs */}
      <Card className="flex-1 overflow-hidden flex flex-col p-4 glass border-blue-500/20">
        <Tabs defaultValue="text" className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-3 lg:grid-cols-6 flex-shrink-0 bg-slate-900/50 border border-cyan-500/20">
            <TabsTrigger value="text" className="text-xs lg:text-sm text-slate-400 data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50">
              <FileText className="w-3 h-3 lg:w-4 lg:h-4 mr-1 lg:mr-2" />
              文本
            </TabsTrigger>
            <TabsTrigger value="tables" className="text-xs lg:text-sm text-slate-400 data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50">
              <Table2 className="w-3 h-3 lg:w-4 lg:h-4 mr-1 lg:mr-2" />
              表格
            </TabsTrigger>
            <TabsTrigger value="formulas" className="text-xs lg:text-sm text-slate-400 data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50">
              <Calculator className="w-3 h-3 lg:w-4 lg:h-4 mr-1 lg:mr-2" />
              公式
            </TabsTrigger>
            <TabsTrigger value="images" className="text-xs lg:text-sm text-slate-400 data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50">
              <Image className="w-3 h-3 lg:w-4 lg:h-4 mr-1 lg:mr-2" />
              图像
            </TabsTrigger>
            <TabsTrigger value="handwritten" className="text-xs lg:text-sm text-slate-400 data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50">
              <PenTool className="w-3 h-3 lg:w-4 lg:h-4 mr-1 lg:mr-2" />
              手写
            </TabsTrigger>
            <TabsTrigger value="performance" className="text-xs lg:text-sm text-slate-400 data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-500/20 data-[state=active]:to-blue-500/20 data-[state=active]:text-cyan-400 data-[state=active]:border-cyan-500/50">
              <BarChart3 className="w-3 h-3 lg:w-4 lg:h-4 mr-1 lg:mr-2" />
              性能
            </TabsTrigger>
          </TabsList>

          {/* Text Recognition Results */}
          <TabsContent value="text" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
                {fullMarkdown ? (
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-cyan-400">解析结果 (Markdown)</h3>
                      <Badge variant="secondary" className="bg-green-500/20 text-green-300 border-green-500/30">
                        完整文档
                      </Badge>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-4 border border-cyan-500/20 max-h-[600px] overflow-y-auto">
                      <pre className="text-slate-300 leading-relaxed whitespace-pre-wrap text-sm">
                        {fullMarkdown}
                      </pre>
                    </div>
                  </div>
                ) : results?.text ? (
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-cyan-400">识别文本</h3>
                      <Badge variant="secondary" className="bg-cyan-500/20 text-cyan-300 border-cyan-500/30">
                        置信度: {results.text.confidence.toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-3 border border-cyan-500/20">
                      <p className="text-slate-300 leading-relaxed whitespace-pre-wrap">
                        {results.text.fullText}
                      </p>
                    </div>

                    <div className="mt-4">
                      <h3 className="text-cyan-400 mb-2">关键词提取</h3>
                      <div className="flex flex-wrap gap-2">
                        {results.text.keywords.map((keyword: string, index: number) => (
                          <Badge key={index} variant="outline" className="bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 border-cyan-500/30 hover:border-cyan-500/50 transition-all">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Tables Results */}
          <TabsContent value="tables" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
                {results.tables && results.tables.length > 0 ? (
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-cyan-500/20 max-h-[600px] overflow-y-auto">
                    <h3 className="text-cyan-400 mb-4">表格识别结果</h3>
                    <div className="space-y-6">
                    {results.tables.map((table: any, index: number) => (
                      <div key={index}>
                        <h3 className="text-cyan-400 mb-2">{table.title}</h3>
                        <div className="border border-cyan-500/20 rounded-lg bg-slate-900/30">
                          <div className="overflow-x-auto">
                            <Table>
                              <TableHeader>
                                <TableRow className="border-cyan-500/20 hover:bg-cyan-500/5">
                                  {table.headers.map((header: string, i: number) => (
                                    <TableHead key={i} className="bg-cyan-500/10 text-cyan-300 border-cyan-500/20 whitespace-nowrap">
                                      {header}
                                    </TableHead>
                                  ))}
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {table.rows.map((row: string[], rowIndex: number) => (
                                  <TableRow key={rowIndex} className="border-cyan-500/20 hover:bg-cyan-500/5">
                                    {row.map((cell: string, cellIndex: number) => (
                                      <TableCell key={cellIndex} className="text-slate-300 border-cyan-500/20 whitespace-nowrap">{cell}</TableCell>
                                    ))}
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                        </div>
                      </div>
                    ))}
                    </div>
                  </div>
                ) : (
                  <Alert className="bg-slate-900/50 border-slate-700 text-slate-400">
                    <AlertDescription>
                      未检测到表格内容
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Formulas Results */}
          <TabsContent value="formulas" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
                <h3 className="text-cyan-400 mb-2">识别的数学公式</h3>
                {results.formulas && results.formulas.length > 0 ? (
                  <div className="space-y-3">
                    {results.formulas.map((formula: any, index: number) => (
                      <div key={index} className="bg-slate-900/50 rounded-lg p-3 border border-purple-500/20">
                        <div className="flex items-start gap-3">
                          <Badge className="mt-1 bg-gradient-to-r from-purple-500/30 to-pink-500/30 text-purple-300 border-purple-500/30">{index + 1}</Badge>
                          <div className="flex-1">
                            <div className="bg-slate-800/50 rounded px-3 py-2 border border-purple-500/20 mb-2 overflow-x-auto">
                              <code className="text-purple-300">{formula.formula}</code>
                            </div>
                            <p className="text-slate-400 text-sm">{formula.description}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Alert className="bg-slate-900/50 border-slate-700 text-slate-400">
                    <AlertDescription>
                      未检测到数学公式
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Images Results */}
          <TabsContent value="images" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
                <div className="bg-slate-900/50 rounded-lg p-4 border border-cyan-500/20 max-h-[600px] overflow-y-auto">
                  <h3 className="text-cyan-400 mb-4">图像识别结果</h3>
                  {results.images && results.images.length > 0 ? (
                    <div className="grid gap-4">
                      {results.images.map((image: any, index: number) => {
                        const imageSrc = image.base64 || (image.path ? `${import.meta.env.VITE_API_URL}/${image.path}` : '');
                        const imageAlt = image.description || image.altText || `图片 ${index + 1}`;

                        return (
                          <div key={index} className="bg-slate-900/50 rounded-lg p-3 border border-blue-500/20">
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <Image className="w-4 h-4 text-blue-400" />
                                <span className="text-slate-200">{image.type}</span>
                              </div>
                              <Badge variant="secondary" className="bg-blue-500/20 text-blue-300 border-blue-500/30">
                                {image.confidence?.toFixed?.(1) || image.confidence}%
                              </Badge>
                            </div>
                            <p className="text-slate-400 mb-3">{imageAlt}</p>
                            {imageSrc ? (
                              <div
                                className="mt-2 rounded-lg overflow-hidden border border-blue-500/20 relative group cursor-pointer"
                                onClick={() => setSelectedImage({ src: imageSrc, alt: imageAlt })}
                              >
                                <img
                                  src={imageSrc}
                                  alt={imageAlt}
                                  className="w-full h-auto max-h-48 object-contain bg-slate-800/50"
                                  onError={(e) => {
                                    console.error('Image load error:', {
                                      base64: !!image.base64,
                                      path: image.path,
                                      src: imageSrc
                                    });
                                    e.currentTarget.parentElement!.innerHTML = `
                                      <div class="w-full h-48 flex items-center justify-center bg-slate-800/50 text-slate-400">
                                        <div class="text-center">
                                          <p>图片加载失败</p>
                                          <p class="text-xs mt-1">${image.path || 'No path'}</p>
                                        </div>
                                      </div>
                                    `;
                                  }}
                                />
                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                  <Maximize2 className="w-8 h-8 text-white" />
                                </div>
                              </div>
                            ) : (
                              <div className="w-full h-48 flex items-center justify-center bg-slate-800/50 text-slate-400 rounded-lg border border-blue-500/20">
                                <p>无图像数据</p>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <Alert className="bg-slate-900/50 border-slate-700 text-slate-400">
                      <AlertDescription>
                        未检测到图像内容
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Handwritten Results */}
          <TabsContent value="handwritten" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
                <h3 className="text-cyan-400 mb-2">手写识别结果</h3>
                {results.handwritten.detected ? (
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-pink-500/20">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <PenTool className="w-4 h-4 text-pink-400" />
                        <span className="text-slate-200">检测到手写内容</span>
                      </div>
                      <Badge variant="secondary" className="bg-pink-500/20 text-pink-300 border-pink-500/30">
                        置信度: {results.handwritten.confidence.toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="bg-slate-800/50 rounded-lg p-3 border border-pink-500/20">
                      <p className="text-slate-300">{results.handwritten.text}</p>
                    </div>
                  </div>
                ) : (
                  <Alert className="bg-slate-900/50 border-slate-700 text-slate-400">
                    <AlertDescription>
                      未检测到手写内容
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Performance Comparison */}
          <TabsContent value="performance" className="flex-1 mt-4 overflow-hidden">
            <ScrollArea className="h-full pr-4">
              <div className="space-y-4">
                <h3 className="text-cyan-400 mb-2">当前模型性能指标</h3>
                <div className="grid grid-cols-3 gap-3">
                  <Card className="p-4 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-cyan-500/30 glow-cyan">
                    <div className="text-center">
                      <p className="text-cyan-300 text-sm mb-1">准确率</p>
                      <p className="text-cyan-400">{results.performance.accuracy}%</p>
                    </div>
                  </Card>
                  <Card className="p-4 bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/30">
                    <div className="text-center">
                      <p className="text-green-300 text-sm mb-1">处理时间</p>
                      <p className="text-green-400">{results.performance.speed}秒</p>
                    </div>
                  </Card>
                  <Card className="p-4 bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-purple-500/30 glow-purple">
                    <div className="text-center">
                      <p className="text-purple-300 text-sm mb-1">内存占用</p>
                      <p className="text-purple-400">{results.performance.memory}MB</p>
                    </div>
                  </Card>
                </div>

                <div className="mt-4">
                  <h3 className="text-cyan-400 mb-2">模型对比说明</h3>
                  <div className="space-y-2">
                    <div className="bg-slate-900/50 rounded-lg p-3 border border-cyan-500/20">
                      <p className="text-cyan-400 mb-1"><strong>MinerU:</strong></p>
                      <p className="text-slate-400 text-sm">高精度文档识别，适合复杂排版和学术文献，准确率最高但处理时间较长。</p>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-3 border border-blue-500/20">
                      <p className="text-blue-400 mb-1"><strong>PaddleOCR-VL:</strong></p>
                      <p className="text-slate-400 text-sm">快速处理引擎，适合批量文档处理，速度最快且内存占用最小。</p>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg p-3 border border-purple-500/20">
                      <p className="text-purple-400 mb-1"><strong>DeepSeek-OCR:</strong></p>
                      <p className="text-slate-400 text-sm">综合性能最佳，在准确率和速度之间取得良好平衡，适合大多数应用场景。</p>
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <h3 className="text-cyan-400 mb-2">源文档对比</h3>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-slate-300 text-sm mb-2">原始文档</p>
                      <div className="bg-slate-900/50 rounded-lg p-4 border border-cyan-500/20 flex items-center justify-center h-32">
                        <div className="text-center text-cyan-400">
                          <FileText className="w-8 h-8 mx-auto mb-1 opacity-50" />
                          <p className="text-xs text-slate-400">{selectedFile?.name}</p>
                        </div>
                      </div>
                    </div>
                    <div>
                      <p className="text-slate-300 text-sm mb-2">OCR 处理结果</p>
                      <div className="bg-slate-800/50 rounded-lg p-3 border border-blue-500/20 h-32 overflow-auto">
                        <p className="text-slate-300 text-xs leading-relaxed">
                          {results.text.fullText.substring(0, 150)}...
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </Card>

      {/* Image Preview Dialog */}
      <Dialog open={!!selectedImage} onOpenChange={() => setSelectedImage(null)}>
        <DialogContent className="max-w-4xl w-full bg-slate-900 border-cyan-500/30">
          <DialogHeader>
            <DialogTitle className="text-cyan-400">{selectedImage?.alt}</DialogTitle>
          </DialogHeader>
          <div className="mt-4">
            {selectedImage && (
              <img
                src={selectedImage.src}
                alt={selectedImage.alt}
                className="w-full h-auto max-h-[70vh] object-contain rounded-lg"
              />
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

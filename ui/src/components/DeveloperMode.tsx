import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface DeveloperModeProps {
  parameters: {
    candidates_k: number;
    alpha: number;
    top_k: number;
    similarity_threshold: number;
  };
  onParametersChange: (params: {
    candidates_k: number;
    alpha: number;
    top_k: number;
    similarity_threshold: number;
  }) => void;
}

const DeveloperMode = ({ parameters, onParametersChange }: DeveloperModeProps) => {
  const updateParameter = (key: keyof typeof parameters, value: number) => {
    onParametersChange({
      ...parameters,
      [key]: value
    });
  };

  return (
    <Card className="developer-panel animate-fade-in">
      <CardHeader className="text-center">
        <CardTitle className="flex items-center justify-center gap-2 arabic-text">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
          وضع المطور
        </CardTitle>
        <CardDescription className="arabic-text">
          إعدادات متقدمة لتحسين نتائج البحث
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-8">
        {/* Candidates K */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <div className="flex flex-col">
              <Label className="text-sm font-medium arabic-text">عدد المرشحين</Label>
              <span className="text-xs text-muted-foreground">Candidates K</span>
            </div>
            <span className="text-sm text-muted-foreground bg-muted px-2 py-1 rounded">
              {parameters.candidates_k}
            </span>
          </div>
          <Slider
            value={[parameters.candidates_k]}
            onValueChange={(value) => updateParameter('candidates_k', value[0])}
            min={50}
            max={500}
            step={25}
            className="w-full"
            dir="ltr"
          />
          <p className="text-xs text-muted-foreground arabic-text">
            عدد النتائج المبدئية للبحث (50-500)
          </p>
        </div>

        {/* Alpha */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <div className="flex flex-col">
              <Label className="text-sm font-medium arabic-text">وزن الترتيب</Label>
              <span className="text-xs text-muted-foreground">Alpha</span>
            </div>
            <span className="text-sm text-muted-foreground bg-muted px-2 py-1 rounded">
              {parameters.alpha.toFixed(1)}
            </span>
          </div>
          <Slider
            value={[parameters.alpha]}
            onValueChange={(value) => updateParameter('alpha', value[0])}
            min={0}
            max={1}
            step={0.1}
            className="w-full"
            dir="ltr"
          />
          <p className="text-xs text-muted-foreground arabic-text">
            توازن بين الدقة والتنوع في النتائج (0-1)
          </p>
        </div>

        {/* Top K */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <div className="flex flex-col">
              <Label className="text-sm font-medium arabic-text">عدد النتائج</Label>
              <span className="text-xs text-muted-foreground">Top K</span>
            </div>
            <span className="text-sm text-muted-foreground bg-muted px-2 py-1 rounded">
              {parameters.top_k}
            </span>
          </div>
          <Slider
            value={[parameters.top_k]}
            onValueChange={(value) => updateParameter('top_k', value[0])}
            min={5}
            max={50}
            step={5}
            className="w-full"
            dir="ltr"
          />
          <p className="text-xs text-muted-foreground arabic-text">
            عدد النتائج النهائية المعروضة (5-50)
          </p>
        </div>

        {/* Similarity Threshold */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <div className="flex flex-col">
              <Label className="text-sm font-medium arabic-text">حد التشابه</Label>
              <span className="text-xs text-muted-foreground">Similarity Threshold</span>
            </div>
            <span className="text-sm text-muted-foreground bg-muted px-2 py-1 rounded">
              {parameters.similarity_threshold.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[parameters.similarity_threshold]}
            onValueChange={(value) => updateParameter('similarity_threshold', value[0])}
            min={0.1}
            max={1}
            step={0.05}
            className="w-full"
            dir="ltr"
          />
          <p className="text-xs text-muted-foreground arabic-text">
            الحد الأدنى للتشابه لعرض النتيجة (0.1-1)
          </p>
        </div>

        {/* Reset Button */}
        <div className="pt-4 border-t border-border">
          <button
            onClick={() => onParametersChange({
              candidates_k: 100,
              alpha: 0.5,
              top_k: 10,
              similarity_threshold: 0.3
            })}
            className="w-full text-sm text-muted-foreground hover:text-foreground transition-colors arabic-text"
          >
            إعادة تعيين القيم الافتراضية
          </button>
        </div>
      </CardContent>
    </Card>
  );
};

export default DeveloperMode;
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { SearchResponse } from '../types/search';

interface SearchResultsProps {
  results: SearchResponse | null;
  isLoading: boolean;
}

const SearchResults = ({ results, isLoading }: SearchResultsProps) => {
  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto">
        <div className="mb-6 text-center">
          <div className="loading-shimmer h-6 w-48 mx-auto rounded mb-2"></div>
          <div className="loading-shimmer h-4 w-32 mx-auto rounded"></div>
        </div>
        
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, index) => (
            <Card key={index} className="result-card">
              <CardContent className="p-6">
                <div className="loading-shimmer h-6 w-3/4 rounded mb-3"></div>
                <div className="loading-shimmer h-4 w-full rounded mb-2"></div>
                <div className="loading-shimmer h-4 w-2/3 rounded mb-4"></div>
                <div className="flex gap-2">
                  <div className="loading-shimmer h-6 w-20 rounded-full"></div>
                  <div className="loading-shimmer h-6 w-24 rounded-full"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (!results) {
    return null;
  }

  return (
    <div className="max-w-6xl mx-auto animate-fade-in">
      {/* Results Header */}
      <div className="mb-8 text-center">
        <h2 className="text-2xl font-bold text-foreground mb-2 arabic-text">
          نتائج البحث عن "{results.query}"
        </h2>
        <div className="flex justify-center items-center gap-4 text-muted-foreground">
          <span className="arabic-text">
            {results.hits_kept.length} نتيجة
          </span>
          <span className="w-1 h-1 bg-muted-foreground rounded-full"></span>
          <span className="arabic-text">
            زمن البحث: {results.total_time.toFixed(2)} ثانية
          </span>
        </div>
        
        {/* Parameters Used (if available) */}
        {results.parameters_used && (
          <div className="mt-4 text-xs text-muted-foreground">
            <details className="inline-block">
              <summary className="cursor-pointer hover:text-foreground transition-colors arabic-text">
                عرض المعاملات المستخدمة
              </summary>
              <div className="mt-2 p-3 bg-muted/50 rounded-lg text-left" dir="ltr">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                  <span>Candidates: {results.parameters_used.candidates_k}</span>
                  <span>Alpha: {results.parameters_used.alpha}</span>
                  <span>Top K: {results.parameters_used.top_k}</span>
                  <span>Threshold: {results.parameters_used.similarity_threshold}</span>
                </div>
              </div>
            </details>
          </div>
        )}
      </div>

      {/* Results Grid */}
      {results.hits_kept.length > 0 ? (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {results.hits_kept.map((hit, index) => (
            <Card key={index} className="result-card group">
              <CardContent className="p-6">
                {/* Service Name */}
                <h3 className="text-lg font-semibold text-foreground mb-3 arabic-text leading-relaxed">
                  {hit.service}
                </h3>
                
                {/* Description */}
                <p className="text-muted-foreground mb-4 arabic-text leading-relaxed text-sm line-clamp-2">
                  {hit.description}
                </p>
                
                {/* Badges */}
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary" className="badge-sector arabic-text">
                    {hit.sector}
                  </Badge>
                  <Badge variant="outline" className="badge-classification arabic-text">
                    {hit.classification}
                  </Badge>
                </div>
                
                {/* Hover indicator */}
                <div className="mt-4 pt-4 border-t border-border opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="flex items-center justify-center text-xs text-muted-foreground">
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="arabic-text">انقر للمزيد من التفاصيل</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="w-24 h-24 mx-auto mb-6 bg-muted rounded-full flex items-center justify-center">
            <svg className="w-12 h-12 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.29-1.009-5.824-2.562M15 6.306a7.962 7.962 0 00-6 0m6 0V3a1 1 0 00-1-1H10a1 1 0 00-1 1v3.306" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-foreground mb-2 arabic-text">
            لم يتم العثور على نتائج
          </h3>
          <p className="text-muted-foreground arabic-text">
            حاول استخدام كلمات مختلفة أو أكثر عمومية
          </p>
        </div>
      )}
    </div>
  );
};

export default SearchResults;
import { useState } from 'react';
import SearchInterface from '../components/SearchInterface';
import DeveloperMode from '../components/DeveloperMode';
import SearchResults from '../components/SearchResults';
import { SearchRequest, SearchResponse } from '../types/search';
import axios from 'axios';
import { toast } from '@/hooks/use-toast';
import namaLogo from '@/assets/nama-logo.svg';

const Index = () => {
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeveloperMode, setIsDeveloperMode] = useState(false);
  const [searchParams, setSearchParams] = useState({
    candidates_k: 100,
    alpha: 0.5,
    top_k: 10,
    similarity_threshold: 0.3
  });

  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      toast({
        title: "خطأ في البحث",
        description: "يرجى إدخال نص للبحث",
        variant: "destructive"
      });
      return;
    }

    setIsLoading(true);
    
    try {
      const searchRequest: SearchRequest = {
        query: query.trim(),
        parameters: isDeveloperMode ? searchParams : undefined
      };

      const response = await axios.post<SearchResponse>(
        `${import.meta.env.VITE_API_URL || 'http://localhost:9001'}/search`,
        searchRequest,
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 30000
        }
      );

      setSearchResults(response.data);
      
      if (response.data.hits_kept.length === 0) {
        toast({
          title: "لم يتم العثور على نتائج",
          description: "حاول استخدام كلمات مختلفة للبحث",
        });
      }
    } catch (error) {
      console.error('Search error:', error);
      toast({
        title: "خطأ في الاتصال بالخادم",
        description: "تعذر الوصول إلى خدمة البحث. يرجى المحاولة لاحقاً",
        variant: "destructive"
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="relative bg-gradient-to-br from-primary to-primary-glow text-primary-foreground overflow-hidden">
        {/* Background Gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-primary-glow/30"></div>
        
        <div className="relative container mx-auto px-4 py-12">
          <div className="text-center max-w-4xl mx-auto">
            <div className="flex items-center justify-center mb-4 animate-fade-in gap-3">
              <h1 className="text-4xl md:text-5xl font-bold arabic-text">
                بحث
              </h1>
              <img
                src={namaLogo}
                alt="نما"
                className="h-16 md:h-20 w-auto filter brightness-0 invert"
              />

            </div>
            
            {/* Decorative Elements */}
            <div className="flex justify-center items-center mt-8 space-x-4 rtl:space-x-reverse">
              <div className="w-2 h-2 bg-secondary rounded-full animate-pulse"></div>
              <div className="w-3 h-3 bg-secondary/70 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-secondary rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Search Interface */}
        <div className="max-w-4xl mx-auto mb-8">
          <SearchInterface
            onSearch={handleSearch}
            isLoading={isLoading}
            isDeveloperMode={isDeveloperMode}
            onToggleDeveloperMode={() => setIsDeveloperMode(!isDeveloperMode)}
          />
          
          {/* Developer Mode Panel */}
          {isDeveloperMode && (
            <div className="mt-6">
              <DeveloperMode
                parameters={searchParams}
                onParametersChange={setSearchParams}
              />
            </div>
          )}
        </div>

        {/* Search Results */}
        {(searchResults || isLoading) && (
          <SearchResults
            results={searchResults}
            isLoading={isLoading}
          />
        )}

      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-border bg-muted/30">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center text-muted-foreground arabic-text">
            <p>© 2025 بحث نما - محرك البحث للخدمات </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
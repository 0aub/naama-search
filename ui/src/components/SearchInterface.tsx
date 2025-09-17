import { useState, KeyboardEvent } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface SearchInterfaceProps {
  onSearch: (query: string) => void;
  isLoading: boolean;
  isDeveloperMode: boolean;
  onToggleDeveloperMode: () => void;
}

const SearchInterface = ({ 
  onSearch, 
  isLoading, 
  isDeveloperMode, 
  onToggleDeveloperMode 
}: SearchInterfaceProps) => {
  const [query, setQuery] = useState('');

  const handleSearch = () => {
    if (query.trim() && !isLoading) {
      onSearch(query);
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Search Box */}
      <div className="search-container">
        <div className="relative">
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="ابحث عن خدمة"
            className="search-input arabic-text text-right pr-16"
            disabled={isLoading}
            dir="rtl"
          />
          
          <Button
            onClick={handleSearch}
            disabled={isLoading || !query.trim()}
            className="search-button"
            size="sm"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
                <span className="text-sm arabic-text">جاري البحث...</span>
              </div>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            )}
          </Button>
        </div>
      </div>

      {/* Developer Mode Toggle */}
      <div className="flex justify-center">
        <Button
          variant="outline"
          onClick={onToggleDeveloperMode}
          className="flex items-center gap-2 arabic-text"
        >
          <svg 
            className={`w-4 h-4 transition-transform ${isDeveloperMode ? 'rotate-180' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          <span>{isDeveloperMode ? 'إخفاء' : 'إظهار'} وضع المطور</span>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
        </Button>
      </div>

      {/* Search Tips */}
      {!isLoading && (
        <div className="text-center">
          <p className="text-sm text-muted-foreground arabic-text">
            نصائح: استخدم كلمات مفتاحية واضحة مثل "زراعة القمح" أو "مكافحة الآفات"
          </p>
        </div>
      )}
    </div>
  );
};

export default SearchInterface;
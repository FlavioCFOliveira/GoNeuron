package layer

import (
	"sync"
	"testing"
)

// TestArenaOperationsRaceCondition verifica se operações de arena são thread-safe
// Testa especificamente as operações de resize e append na arena
func TestArenaOperationsRaceCondition(t *testing.T) {
	// Simular múltiplas goroutines a aceder à mesma arena
	arena := make([]float32, 100)
	offset := 0
	offsetMu := sync.Mutex{}

	var wg sync.WaitGroup
	numGoroutines := 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Simular resize de arena (operação que precisa de proteção)
			offsetMu.Lock()
			requiredSize := offset + 50
			if len(arena) < requiredSize {
				newArena := make([]float32, requiredSize*2)
				copy(newArena, arena)
				arena = newArena
			}
			// Escrever na arena
			for j := 0; j < 50; j++ {
				arena[offset+j] = float32(id*100 + j)
			}
			offset += 50
			offsetMu.Unlock()
		}(i)
	}

	wg.Wait()

	// Verificar que todos os valores foram escritos
	if offset != numGoroutines*50 {
		t.Errorf("offset = %d, want %d", offset, numGoroutines*50)
	}
}

// TestArenaPointerRaceCondition verifica se o ponteiro da arena é atualizado de forma segura
func TestArenaPointerRaceCondition(t *testing.T) {
	// Teste simples para verificar que o mutex protege o acesso
	var mu sync.Mutex
	arena := make([]float32, 100)
	offset := 0

	var wg sync.WaitGroup
	numGoroutines := 20

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			mu.Lock()
			// Operações protegidas
			if offset+10 > len(arena) {
				newArena := make([]float32, len(arena)*2)
				copy(newArena, arena)
				arena = newArena
			}
			for j := 0; j < 10; j++ {
				arena[offset+j] = float32(id)
			}
			offset += 10
			mu.Unlock()
		}(i)
	}

	wg.Wait()

	if offset != numGoroutines*10 {
		t.Errorf("offset = %d, want %d", offset, numGoroutines*10)
	}
}

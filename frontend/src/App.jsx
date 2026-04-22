import { useState } from 'react'
import { Container, Toast, ToastContainer } from 'react-bootstrap'
import { useSettings } from './lib/useSettings'
import RAGPage from './RAGPage'
import './App.css'

function App() {
  const S = useSettings()
  const [toasts, setToasts] = useState([])

  const toast = (variant, message) => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, variant, message }])
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 5000)
  }

  return (
    <Container fluid className="App">
      <h1>📄 PDF RAG Application</h1>
      <RAGPage S={S} toast={toast} />
      
      <ToastContainer position="top-end" className="p-3">
        {toasts.map(t => (
          <Toast
            key={t.id}
            bg={t.variant}
            onClose={() => setToasts(prev => prev.filter(x => x.id !== t.id))}
            autohide
          >
            <Toast.Header>
              <strong className="me-auto">Notification</strong>
            </Toast.Header>
            <Toast.Body className={t.variant === 'danger' ? 'text-white' : ''}>
              {t.message}
            </Toast.Body>
          </Toast>
        ))}
      </ToastContainer>
    </Container>
  )
}

export default App

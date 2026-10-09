# Plan: mejoras de la interfaz y almacenamiento de reuniones

## Objetivo

Hacer más visible el estado de grabación en Windows, actualizar la interfaz a
un tema claro y reducir el espacio ocupado por el audio sin poner en riesgo una
captura ni perder reuniones si falla la conversión.

## Estado actual relevante

- La aplicación usa Tkinter y una ventana principal con pestañas.
- La interfaz actual tiene un tema oscuro y ya carga `icon.ico`.
- La captura produce `meeting_audio.wav` dentro de la carpeta de la reunión.
- La biblioteca calcula la duración leyendo ese WAV. El procesamiento posterior
  también se inicia a partir del audio capturado.
- El estado de la interfaz se actualiza mediante mensajes y callbacks de Tkinter;
  la captura y el procesamiento no deben bloquear el hilo de la ventana.
- El árbol de trabajo ya contiene otros cambios sin commit. Este plan no los
  modifica.

## Propuesta de implementación

### 1. Indicador compacto al minimizar

1. Añadir una vista compacta de estado vinculada a los estados existentes de la
   reunión, no un segundo mecanismo de grabación.
2. Al minimizar la ventana principal durante una grabación, mostrar un pequeño
   indicador flotante con estado visible, tiempo transcurrido y una acción para
   volver a la ventana principal.
3. Mantener toda interacción con Tkinter en el hilo de UI y consumir el estado
   publicado por el flujo actual de mensajes.
4. Cerrar u ocultar el indicador al detener la captura, al cerrar la app o al
   restaurar la ventana principal. No mostrarlo como si se grabara durante el
   procesamiento posterior.
5. Si falla la inicialización o actualización del indicador, registrar el error
   y conservar la ventana principal y la captura como fuente fiable del estado.

### 2. Tema claro y tipografía

1. Cambiar los estilos ttk/Tkinter a una paleta clara coherente, con contraste
   suficiente en texto, botones, pestañas, tablas, estados y vista previa.
2. Usar Segoe UI como fuente preferida de Windows y tamaños/pesos consistentes.
3. Mantener la jerarquía actual de controles, textos y acciones; limitar el
   alcance a presentación y legibilidad.
4. Comprobar tamaños de ventana, tablas, vista previa Markdown, estados de error
   y controles deshabilitados.

### 3. Conversión de WAV a MP3 al finalizar todo el proceso

1. Mantener WAV como formato de trabajo durante la captura y todas las etapas
   posteriores: transcripción, resumen e informes opcionales. No convertir ni
   borrar el WAV mientras una etapa todavía lo utiliza.
2. Ejecutar la conversión únicamente cuando el procesamiento de la reunión haya
   terminado por completo, con éxito o con un resultado final registrado. La
   conversión es una optimización de almacenamiento, no una dependencia de la
   transcripción ni del resumen.
3. Convertir a un archivo temporal en la misma carpeta y validarlo antes de
   promoverlo a `meeting_audio.mp3`. Validar que sea legible y que la duración
   sea razonablemente consistente con el WAV original.
4. Borrar el WAV solo después de validar el MP3. Si falla el codec, la
   conversión o la validación, conservar el WAV, dejar constancia clara del
   error y permitir reintentar la optimización sin reprocesar la reunión.
5. Si la transcripción o el resumen fallan, guardar ese resultado de error y
   finalizar el flujo; luego se puede intentar la conversión porque el WAV ya
   no está en uso. Nunca condicionar la validez del resultado de reunión a que
   la optimización MP3 haya funcionado.
6. Actualizar la biblioteca para obtener la duración de MP3 nuevos y WAV
   heredados. Mantener el soporte de WAV para reuniones existentes y reuniones
   cuya conversión haya fallado.
7. Definir y documentar el método de conversión y su disponibilidad en Windows
   (por ejemplo, FFmpeg instalado o distribuido con la aplicación) antes de
   elegir dependencias o modificar el empaquetado.

### 4. Indicador de aplicación en la barra de tareas de Windows

1. Verificar que la ventana principal y el ejecutable distribuido se agrupen
   bajo el icono correcto en la barra de tareas, reutilizando `icon.ico`.
2. Evitar estilos de ventana que la excluyan de la barra de tareas y comprobar
   minimizar/restaurar, ventana compacta y varias instancias si están admitidas.
3. Tratar la línea inferior de “aplicación en ejecución” como un indicador que
   dibuja Windows. La aplicación puede mantener una ventana correctamente
   asociada al icono, pero no imponer el color gris: depende de Windows, el tema
   y la configuración del usuario.

## Riesgos de desestabilizar la grabación

| Riesgo | Probabilidad / impacto | Mitigación | Verificación |
|---|---|---|---|
| La conversión se ejecuta mientras otro proceso todavía lee el WAV | Baja / Alta | Convertir solo tras completar todas las etapas y cerrar lectores/archivos; hacer la optimización fuera del hilo de UI/audio | Reunión completa con resumen activado/desactivado y confirmación de que la conversión empieza al final |
| El WAV se borra aunque el MP3 esté truncado o no sea utilizable | Baja / Crítica | Archivo temporal, validación de lectura/duración y borrado del WAV únicamente tras validar el MP3 | Simular salida vacía, MP3 corrupto y error de conversión; el WAV debe permanecer |
| El fallo de conversión altera el resultado de la reunión o impide consultar su transcripción | Baja / Alta | Tratar conversión como postproceso no crítico; registrar el fallo sin revertir ni bloquear el resultado de transcripción/resumen | Simular codec ausente después de procesar; reunión y documentos deben seguir disponibles |
| La biblioteca no reconoce el formato de audio retenido | Media / Alta | Leer duración de MP3 y WAV heredados o retenidos tras un error | Pruebas de WAV y MP3, carpeta incompleta y reunión histórica |
| No está disponible el codec en una instalación de Windows | Media / Alta | Comprobar el mecanismo de distribución/instalación y fallar explícitamente con fallback al WAV, sin borrar el origen | Probar instalación limpia sin el codec y con el codec disponible |
| La ventana compacta provoca estados de UI inconsistentes o duplica acciones | Media / Media | Una única fuente de estado; indicador controlado desde la ventana raíz y acciones idempotentes | Minimizar/restaurar repetidamente durante grabación, detención y error |
| La ventana compacta oculta el botón de la barra de tareas o cambia la agrupación | Media / Media | Validar estilos y asociación de ventana con el icono en el Windows objetivo | Prueba manual de barra de tareas con el paquete distribuido |
| El tema claro reduce contraste o rompe layouts existentes | Baja / Media | Revisar todos los estados y componentes ttk/Tkinter, no solo el fondo | Revisión manual en resolución objetivo y pruebas existentes de UI |

## Orden de trabajo recomendado

1. Registrar línea base: ejecutar la suite y una prueba manual corta de captura;
   confirmar WAV, duración, transcripción y comportamiento de minimizar.
2. Aplicar el tema claro en un cambio visual aislado.
3. Añadir el indicador compacto sin cambiar el ciclo de vida del grabador.
4. Implementar la optimización WAV→MP3 como una etapa posterior a todo el
   procesamiento, con WAV heredados todavía soportados y eliminación
   condicionada por validaciones.
5. Validar icono y presencia en la barra de tareas usando el artefacto real de
   Windows, además de probar la ventana compacta.
6. Ejecutar suite completa, pruebas de fallos de conversión y una reunión real
   de principio a fin antes de dar por terminados los cambios.

## Criterio de aceptación

- Minimizar durante una captura deja visible un indicador compacto con estado y
  tiempo; restaurar o detener la reunión no deja indicadores huérfanos.
- Los controles existentes siguen funcionando con una paleta clara y legible.
- La grabación y todas las etapas de procesamiento terminan usando el WAV; la
  conversión a MP3 empieza solo al finalizar el flujo completo.
- Una conversión válida produce MP3; el WAV solo se elimina después de validar
  el MP3.
- Un error de codec/conversión conserva el WAV, pero no invalida ni oculta la
  transcripción o el resumen ya generados; el error de optimización se comunica
  con claridad.
- La lista muestra reuniones MP3 nuevas y WAV antiguas con su duración.
- En Windows, la app aparece agrupada bajo su icono en la barra de tareas; el
  aspecto exacto del indicador de ejecución queda a cargo del sistema.

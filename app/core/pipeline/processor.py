    async def transcribe_speech(
        self,
        audio_content: bytes,
        language: Optional[str] = None,
        detect_language: bool = False,
        model_id: Optional[str] = None,
        options: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe speech from audio.
        
        Args:
            audio_content: Audio content as bytes
            language: Language code
            detect_language: Whether to detect language
            model_id: Optional model ID
            options: Additional options
            user_id: Optional user ID for tracking
            request_id: Optional request ID for tracking
            
        Returns:
            Dict with transcription results
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.stt_pipeline:
            logger.warning("STT pipeline not initialized, initializing now")
            await self._initialize_stt_pipeline()
            
        start_time = time.time()
        
        # Prepare options dict
        if options is None:
            options = {}
            
        options["user_id"] = user_id
        options["request_id"] = request_id
        options["model_id"] = model_id
        
        # Transcribe using STT pipeline
        transcription_result = await self.stt_pipeline.transcribe(
            audio_content=audio_content,
            language=language,
            detect_language=detect_language,
            options=options
        )
        
        # Add process time
        process_time = time.time() - start_time
        if isinstance(transcription_result, dict):
            transcription_result["process_time"] = process_time
            
        # Add request information
        if isinstance(transcription_result, dict):
            transcription_result["request_id"] = request_id
            
        return transcription_result

    async def get_stt_languages(self) -> Dict[str, Any]:
        """
        Get supported languages for speech recognition.
        
        Returns:
            Dict with supported languages
        """
        if not self.initialized:
            await self.initialize()
            
        if not self.stt_pipeline:
            logger.warning("STT pipeline not initialized, initializing now")
            await self._initialize_stt_pipeline()
            
        return await self.stt_pipeline.get_supported_languages()